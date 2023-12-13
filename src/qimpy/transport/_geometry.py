from __future__ import annotations
from typing import Sequence, Union, Any, Optional
from collections import namedtuple, defaultdict

import numpy as np
import torch
from xml.dom import minidom
from svg.path import parse_path, CubicBezier, Line, Close

from qimpy import TreeNode, rc
from qimpy.io import CheckpointPath
from qimpy.transport.advect._advect import Advect


PatchSet = namedtuple("PatchSet", ["vertices", "edges", "quads", "adjacency"])


class CubicSpline:
    spline_params: torch.Tensor  # 4 x 2 tensor
    neighbor_edge: CubicSpline
    n_points: int

    def __init__(self, spline_params, neighbor_edge=None, n_points=64):
        self.spline_params = spline_params
        self.n_points = n_points
        self.neighbor_edge = None

    def __repr__(self):
        numpy_spline = self.spline_params.numpy()
        return f"{numpy_spline[0, :]} -> {numpy_spline[-1, :]} (neighbor: {self.neighbor_edge is not None})"

    def points(self):
        assert len(self.spline_params) == 4
        t = np.linspace(0.0, 1.0, self.n_points + 1)[:, None]
        t_bar = 1.0 - t
        # Evaluate cubic spline by De Casteljau's algorithm:
        control_points = self.spline_params.to(rc.cpu).numpy()
        result = control_points[:, None, :]
        for i_iter in range(len(self.spline_params) - 1):
            result = result[:-1] * t_bar + result[1:] * t
        return result[0]


class BicubicPatch:
    """Transformation based on cubic spline edges."""

    control_points: torch.Tensor  #: Control point coordinates (4 x 4 x 2)

    def __init__(self, boundary: torch.Tensor):
        """Initialize from 12 x 2 coordinates of control points on perimeter."""
        control_points = torch.empty((4, 4, 2), device=rc.device)
        # Set boundary control points:
        control_points[:, 0] = boundary[:4]
        control_points[-1, 1:] = boundary[4:7]
        control_points[:-1, -1] = boundary[7:10].flipud()
        control_points[0, 1:-1] = boundary[10:12].flipud()

        # Set internal control points based on parallelogram completion:
        def complete_parallelogram(
            v: torch.Tensor, i0: int, j0: int, i1: int, j1: int
        ) -> None:
            v[i1, j1] = v[i0, j1] + v[i1, j0] - v[i0, j0]

        complete_parallelogram(control_points, 0, 0, 1, 1)
        complete_parallelogram(control_points, 3, 0, 2, 1)
        complete_parallelogram(control_points, 0, 3, 1, 2)
        complete_parallelogram(control_points, 3, 3, 2, 2)
        self.control_points = control_points

    def __call__(self, Qfrac: torch.Tensor) -> torch.Tensor:
        """Define mapping from fractional mesh to Cartesian coordinates."""
        return torch.einsum(
            "uvi, u..., v... -> ...i",
            self.control_points,
            cubic_bernstein(Qfrac[..., 0]),
            cubic_bernstein(Qfrac[..., 1]),
        )


def cubic_bernstein(t: torch.Tensor) -> torch.Tensor:
    """Return basis of cubic Bernstein polynomials."""
    t_bar = 1.0 - t
    return torch.stack(
        (t_bar**3, 3.0 * (t_bar**2) * t, 3.0 * t_bar * (t**2), t**3)
    )


def weld_points(coords: torch.Tensor, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Weld `coords` within tolerance `tol`, returning indices and unique coordinates.
    Here, coords has dimensions (..., d), where d is the dimension of space.
    The first output is a flat list of unique welded vertices of shape (N_uniq, d).
    The second output contains indices into this unique list of dimensions (...)."""
    coords_flat = coords.flatten(end_dim=-2)
    distances = (coords_flat[:, None] - coords_flat[None]).norm(dim=-1)
    equiv_index = torch.where(distances < tol, 1, 0).argmax(dim=1)
    _, inverse, counts = torch.unique(
        equiv_index, return_inverse=True, return_counts=True
    )
    # Compute the centroid of each set of equivalent vertices:
    coords_uniq = torch.zeros((len(counts), coords_flat.shape[-1]), device=rc.device)
    coords_uniq.index_add_(0, inverse, coords_flat)  # sum equivalent coordinates
    coords_uniq *= (1.0 / counts)[:, None]  # convert to mean
    return coords_uniq, inverse.view(coords.shape[:-1])


PATCH_SIDES: int = 4  #: Support only quad-patches (implicitly required throughout)


def parse_style(style_str: str):
    return {
        prop: value for prop, value in [cmd.split(":") for cmd in style_str.split(";")]
    }


def get_splines(svg_file: str) -> torch.Tensor:
    doc = minidom.parse(svg_file)
    svg_elements = []
    styles = []
    svg_paths = doc.getElementsByTagName("path")

    # Concatenate segments from all paths in SVG file, and parse associated styles
    for path in svg_paths:
        paths = parse_path(path.getAttribute("d"))
        svg_elements.extend(paths)
        styles.extend(len(paths) * [parse_style(path.getAttribute("style"))])

    def segment_to_tensor(segment):
        if isinstance(segment, CubicBezier):
            control1, control2 = segment.control1, segment.control2
        # Both Line and Close can produce linear segments
        elif isinstance(segment, (Line, Close)):
            # Generate a spline from a linear segment
            disp_third = (segment.end - segment.start) / 3.0
            control1 = segment.start + disp_third
            control2 = segment.start + 2 * disp_third
        else:
            raise ValueError("All segments must be cubic splines or lines")
        return torch.view_as_real(
            torch.tensor(
                [segment.start, control1, control2, segment.end],
                device=rc.device,
            )
        )

    # Ignore all elements that are not lines or cubic splines (essentially ignore moves)
    # In the future we may want to throw an error for unsupported segments
    # (e.g. quadratic splines)
    splines = []
    colors = []
    for i, segment in enumerate(svg_elements):
        if isinstance(segment, (Line, Close, CubicBezier)):
            splines.append(segment_to_tensor(segment))
            colors.append(styles[i]["stroke"])
    splines = torch.stack(splines)

    return splines, colors


def edge_sequence(cycle):
    return list(zip(cycle[:-1], cycle[1:])) + [(cycle[-1], cycle[0])]


class SVGParser:
    def __init__(self, svg_file, epsilon=0.005):
        self.splines, self.colors = get_splines(svg_file)
        self.vertices, self.edges = weld_points(self.splines[:, (0, -1)], tol=epsilon)
        self.edges_lookup = {
            (edge[0], edge[1]): ind for ind, edge in enumerate(self.edges.tolist())
        }

        self.cycles = []
        self.find_cycles()

        self.patches = []

        verts = self.vertices.clone().detach().to(device=rc.device)
        edges = torch.tensor([], dtype=torch.int, device=rc.device)
        quads = torch.tensor([], dtype=torch.int, device=rc.device)

        control_pt_lookup = {}
        quad_edges = {}
        color_adj = {}

        color_pairs = defaultdict(list)
        for i, color in enumerate(self.colors):
            # Ignore black edges
            if color != "#000000":
                color_pairs[color].append(i)

        # Only include pairs, exclude all others
        color_pairs = {key: val for key, val in color_pairs.items() if len(val) == 2}

        # Now build the patches, ensuring each spline goes along
        # the direction of the cycle
        for cycle in self.cycles:
            cur_quad = []
            for edge in edge_sequence(cycle):
                # Edges lookup reflects the original ordering of the edges
                # if an edge's order doesn't appear in here, it needs to be flipped
                if edge not in self.edges_lookup:
                    spline = torch.flip(
                        self.splines[self.edges_lookup[edge[::-1]]], dims=[0]
                    )
                    color = self.colors[self.edges_lookup[edge[::-1]]]
                else:
                    spline = self.splines[self.edges_lookup[edge]]
                    color = self.colors[self.edges_lookup[edge]]
                cp1 = tuple(spline[1].tolist())
                cp2 = tuple(spline[2].tolist())
                # Get control points from spline and add to vertices
                # Ensure that control points are unique by lookup dict
                if cp1 not in control_pt_lookup:
                    verts = torch.cat((verts, spline[1:3]), 0)
                    control_pt_lookup[cp1] = verts.shape[0] - 2
                    control_pt_lookup[cp2] = verts.shape[0] - 1
                edges = torch.cat(
                    (
                        edges,
                        torch.tensor(
                            [
                                [
                                    edge[0],
                                    control_pt_lookup[cp1],
                                    control_pt_lookup[cp2],
                                    edge[1],
                                ]
                            ]
                        ),
                    ),
                    0,
                )
                cur_quad.append(edges.shape[0] - 1)
                quad_edges[edge] = (int(quads.shape[0]), int(len(cur_quad) - 1))
                if color in color_pairs:
                    color_adj[edge] = color
            quads = torch.cat((quads, torch.tensor([cur_quad])), 0)
        adjacency = -1 * torch.ones(
            [len(quads), PATCH_SIDES, 2], dtype=torch.int, device=rc.device
        )
        for edge, adj in quad_edges.items():
            quad, edge_ind = adj
            # Handle inner adjacency
            if edge[::-1] in quad_edges:
                adjacency[quad, edge_ind, :] = torch.tensor(
                    list(quad_edges[edge[::-1]])
                )

            # Handle color adjacency
            if edge in color_adj:
                color = color_adj[edge]
                # N^2 lookup, fine for now
                for other_edge, other_color in color_adj.items():
                    if other_color == color and edge != other_edge:
                        adjacency[quad, edge_ind, :] = torch.tensor(
                            list(quad_edges[other_edge])
                        )

        self.patch_set = PatchSet(verts, edges, quads, adjacency)

    # Determine whether a cycle goes counter-clockwise or clockwise
    # (Return 1 or -1 respectively)
    def cycle_handedness(self, cycle):
        cycle_vertices = [self.vertices[j] for j in cycle]
        edges = edge_sequence(cycle_vertices)
        handed_sum = 0.0
        for v1, v2 in edges:
            handed_sum += (v2[0] - v1[0]) / (v2[1] + v1[1])
        # NOTE: SVG uses a left-handed coordinate system
        return np.sign(handed_sum)

    def add_cycle(self, cycle):
        # Add a cycle if it is unique

        def unique(path):
            return path not in self.cycles

        def normalize_cycle_order(cycle):
            min_index = cycle.index(min(cycle))
            return cycle[min_index:] + cycle[:min_index]

        new_cycle = normalize_cycle_order(cycle)
        # Check both directions
        if unique(new_cycle) and unique(normalize_cycle_order(new_cycle[::-1])):
            self.cycles.append(new_cycle)

    def find_cycles(self):
        # Graph traversal using recursion
        def cycle_search(cycle, depth=1):
            # Don't look for cycles that exceed a single patch (limit recursion depth)
            if depth > PATCH_SIDES:
                return
            start_vertex = cycle[-1]
            for edge in self.edges:
                if start_vertex in edge:
                    next_vertex = int(edge[1] if edge[0] == start_vertex else edge[0])
                    if next_vertex not in cycle:
                        cycle_search(cycle + [next_vertex], depth=depth + 1)
                    elif len(cycle) > 2 and next_vertex == cycle[0]:
                        self.add_cycle(cycle)

        # Search for cycles from each starting vertex
        for first_vertex in range(len(self.vertices)):
            cycle_search([first_vertex])

        # Make sure each cycle goes counter-clockwise
        self.cycles = [
            cycle if self.cycle_handedness(cycle) > 0 else cycle[::-1]
            for cycle in self.cycles
        ]


class Geometry(TreeNode):
    """Geometry specification."""

    vertices: torch.Tensor
    edges: torch.Tensor
    quads: torch.Tensor
    adjacency: torch.Tensor

    # v_F and N_theta should eventually be material paramteters
    def __init__(
        self,
        *,
        svg_file: str,
        N: tuple[int, int],
        N_theta: int,
        v_F: torch.Tensor,
        # For now, we are testing horizontal or diagonal advection
        diag: bool,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        svg_file
            :yaml:`Path to an SVG file containing the input geometry.
        """
        super().__init__()

        svg_parser = SVGParser(svg_file)
        self.patch_set = svg_parser.patch_set
        self.patches = []
        # Build an advect object for each quad
        for i_quad, quad in enumerate(self.patch_set.quads):
            boundary = []
            for edge in self.patch_set.edges[quad]:
                for coord in self.patch_set.vertices[edge][:-1]:
                    boundary.append(coord.tolist())
            boundary = torch.tensor(boundary, device=rc.device)
            transformation = BicubicPatch(boundary=boundary)

            # Initialize velocity and transformation based on first patch:
            if i_quad == 0:
                origin = transformation(torch.zeros((1, 2), device=rc.device))
                Rbasis = (transformation(torch.eye(2, device=rc.device)) - origin).T
                delta_Qfrac = torch.tensor(
                    [-1.0, -1.0] if diag else [1.0, 0.0], device=rc.device
                )
                delta_q = delta_Qfrac @ Rbasis.T

                # Initialize velocities (eventually should be in Material):
                init_angle = torch.atan2(delta_q[1], delta_q[0]).item()
                dtheta = 2 * np.pi / N_theta
                theta = torch.arange(N_theta, device=rc.device) * dtheta + init_angle
                v = v_F * torch.stack([theta.cos(), theta.sin()], dim=-1)

            new_patch = Advect(transformation=transformation, v=v, N=N)
            new_patch.origin = origin
            new_patch.Rbasis = Rbasis
            self.patches.append(new_patch)

    def apply_boundaries(self, patch_ind, patch, rho) -> torch.Tensor:
        """Apply all boundary conditions to `rho` and produce ghost-padded version."""
        non_ghost = patch.non_ghost
        ghost_l = patch.ghost_l
        ghost_r = patch.ghost_r
        out = torch.zeros(patch.rho_padded_shape, device=rc.device)
        out[non_ghost, non_ghost] = rho

        patch_adj = self.patch_set.adjacency[patch_ind]

        # TODO: handle reflecting boundaries
        # For now they will be sinks (hence pass)

        # This logic is not correct yet (flips/transposes)
        # Check if each edge is reflecting, otherwise handle
        # ghost zone communication (16 separate cases)
        if int(patch_adj[0, 0]) == -1:
            pass
        else:
            other_patch_ind, other_edge = patch_adj[0, :].tolist()
            other_patch = self.patches[other_patch_ind]
            if other_edge == 0:
                ghost_area = torch.flip(
                    other_patch.rho_prev[:, ghost_l],
                    dims=(1,),
                )
            if other_edge == 1:
                ghost_area = torch.flip(
                    torch.transpose(other_patch.rho_prev[ghost_r], 0, 1),
                    dims=(0, 1),
                )
            if other_edge == 2:
                ghost_area = other_patch.rho_prev[:, ghost_r]
            if other_edge == 3:
                ghost_area = torch.flip(
                    torch.transpose(other_patch.rho_prev[ghost_l], 0, 1),
                    dims=(0,),
                )
            out[non_ghost, ghost_l] = ghost_area

        if int(patch_adj[1, 0]) == -1:
            pass
        else:
            other_patch_ind, other_edge = patch_adj[1, :].tolist()
            other_patch = self.patches[other_patch_ind]
            if other_edge == 0:
                ghost_area = torch.flip(
                    torch.transpose(other_patch.rho_prev[:, ghost_l], 0, 1),
                    dims=(1,),
                )
            if other_edge == 1:
                ghost_area = torch.flip(other_patch.rho_prev[ghost_r], dims=(0, 1))
            if other_edge == 2:
                ghost_area = torch.transpose(other_patch.rho_prev[:, ghost_r], 0, 1)
            if other_edge == 3:
                ghost_area = other_patch.rho_prev[ghost_l]
            out[ghost_r, non_ghost] = ghost_area

        if int(patch_adj[2, 0]) == -1:
            pass
        else:
            other_patch_ind, other_edge = patch_adj[2, :].tolist()
            other_patch = self.patches[other_patch_ind]
            if other_edge == 0:
                ghost_area = other_patch.rho_prev[:, ghost_l]
            if other_edge == 1:
                ghost_area = torch.flip(
                    torch.transpose(other_patch.rho_prev[ghost_r], 0, 1),
                    dims=(0,),
                )
            if other_edge == 2:
                ghost_area = torch.flip(other_patch.rho_prev[:, ghost_r], dims=[0, 1])
            if other_edge == 3:
                ghost_area = torch.flip(
                    torch.transpose(other_patch.rho_prev[ghost_l], 0, 1),
                    dims=(0,),
                )
            out[non_ghost, ghost_r] = ghost_area

        if int(patch_adj[3, 0]) == -1:
            pass
        else:
            other_patch_ind, other_edge = patch_adj[3, :].tolist()
            other_patch = self.patches[other_patch_ind]
            if other_edge == 0:
                ghost_area = other_patch.rho_prev[:, ghost_l]
            if other_edge == 1:
                ghost_area = other_patch.rho_prev[ghost_r]
            if other_edge == 2:
                ghost_area = torch.flip(
                    torch.transpose(other_patch.rho_prev[:, ghost_r], 0, 1),
                    dims=(0, 1),
                )
            if other_edge == 3:
                ghost_area = torch.flip(other_patch.rho_prev[ghost_l], dims=(0, 1))
            out[ghost_l, non_ghost] = ghost_area

        return out

    # Geometry level time step
    def time_step(self):
        for patch in self.patches:
            patch.rho_prev = patch.rho.detach().clone()
        for i, patch in enumerate(self.patches):
            rho_half = patch.rho + patch.drho(
                0.5 * patch.dt, self.apply_boundaries(i, patch, patch.rho_prev)
            )
            patch.rho += patch.drho(patch.dt, self.apply_boundaries(i, patch, rho_half))


def _make_check_tensor(
    data: Union[Sequence[Sequence[Any]], np.ndarray, torch.Tensor],
    dims: Sequence[int],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    result = torch.tensor(data, device=rc.device, dtype=dtype)
    assert len(result.shape) == len(dims)
    for result_shape_i, dim_i in zip(result.shape, dims):
        if dim_i >= 0:
            assert result_shape_i == dim_i
    return result


def equivalence_classes(pairs: torch.Tensor) -> torch.Tensor:
    """Given Npair x 2 array of index pairs that are equivalent,
    compute equivalence class numbers for each original index."""
    # Construct adjacency matrix:
    N = pairs.max() + 1
    i_pair, j_pair = pairs.T
    adjacency_matrix = torch.eye(N, device=rc.device)
    adjacency_matrix[i_pair, j_pair] = 1.0
    adjacency_matrix[j_pair, i_pair] = 1.0

    # Expand to indirect neighbors by repeated multiplication:
    n_non_zero_prev = torch.count_nonzero(adjacency_matrix)
    for i_mult in range(N):
        adjacency_matrix = adjacency_matrix @ adjacency_matrix
        n_non_zero = torch.count_nonzero(adjacency_matrix)
        if n_non_zero == n_non_zero_prev:
            break  # highest-degree connection reached
        n_non_zero_prev = n_non_zero

    # Find first non-zero entry of above (i.e. first equivalent index):
    is_first = torch.logical_and(
        adjacency_matrix.cumsum(dim=1) == adjacency_matrix, adjacency_matrix != 0.0
    )
    first_index = torch.nonzero(is_first)[:, 1]
    assert len(first_index) == N
    return torch.unique(first_index, return_inverse=True)[1]  # minimal class indices
