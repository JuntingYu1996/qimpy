from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import numpy as np

from qimpy import log
from qimpy.math import ceildiv
from . import QuadSet


@dataclass
class SubQuadSet:
    """Specifies subdivision of a QuadSet (for flexibility in MPI parallelization)."""

    quad_index: np.ndarray  #: original quad index that each sub-quad corresponds to
    grid_start: np.ndarray  #: starting grid index of sub-quad (NsubQuads x 2)
    grid_stop: np.ndarray  #: stopping grid index of sub-quad (NsubQuads x 2)
    adjacency: np.ndarray  #: NsubQuads x 4 x 2, analogous to `QuadSet.adjacency`
    has_apertures: np.ndarray  #: NsubQuads x 4, analogous to `QuadSet.has_apertures`


def divided_count(quad_set: QuadSet, grid_size_max: int) -> tuple[int, float]:
    """Calculate number of quads if `quad_set` is divided such that
    the maximum sample count of any resulting quad is `grid_size`.
    Also return the percentage imbalance between subdivided quad sizes."""
    n_splits = ceildiv(quad_set.grid_size, grid_size_max)  # 2 splits for each quad
    n_divided_quads = n_splits.prod(axis=1).sum(axis=0)
    split_sizes = quad_set.grid_size / n_splits  # approximate: assuming equal splits
    split_counts = split_sizes.prod(axis=1)
    imbalance_percent = 100.0 * split_counts.std() / split_counts.mean()
    return n_divided_quads, imbalance_percent


def select_division(quad_set: QuadSet, n_processes: int) -> int:
    """Select `grid_size_max` suitable for division on `n_processes`.
    Reports all cases considered to guide selection of parallelization."""
    # Start with existing sizes, attempting to make domains squarer
    grid_size_max_list = np.unique(quad_set.grid_size)

    # Expand list with smaller entries if needed
    n_max = divided_count(quad_set, grid_size_max_list[0])[0]  # type:ignore
    needed_expansion = 4 * n_processes / n_max  # check quad counts till 4 n_processes
    if needed_expansion > 1.0:
        log_spacing = 0.2
        scale_factors = np.exp(
            -log_spacing - np.arange(0, 0.5 * np.log(needed_expansion), log_spacing)
        )
        additional_sizes = np.maximum(
            1, np.round(grid_size_max_list[0] * scale_factors).astype(int)
        )
        grid_size_max_list = np.unique(
            np.concatenate((grid_size_max_list, additional_sizes))
        )

    # Fill any large geometric gaps in above list:
    gap_ratios = grid_size_max_list[1:] / grid_size_max_list[:-1]
    for i_gap, gap_ratio in enumerate(gap_ratios):
        if gap_ratio > 1.5:
            log_spacing = 0.2
            scale_factors = np.exp(np.arange(0, np.log(gap_ratio), log_spacing))
            additional_sizes = np.round(grid_size_max_list[i_gap] * scale_factors)
            grid_size_max_list = np.unique(
                np.concatenate((grid_size_max_list, additional_sizes.astype(int)))
            )

    # Check and report candidates:
    log.info("\n--- Quad subdivision candidates ---")
    log.info("grid_size_max  n_quads  size_imbalance number_imbalance")
    best_grid_size_max = 0
    best_score = (np.inf,) * 3
    for grid_size_max in grid_size_max_list[::-1]:
        n_quads, size_imbalance = divided_count(quad_set, grid_size_max)
        n_quads_each = ceildiv(n_quads, n_processes)
        number_imbalance = 100.0 * (1.0 - n_quads / (n_quads_each * n_processes))
        score = (number_imbalance, size_imbalance, n_quads)
        if score < best_score:
            best_score = score
            best_grid_size_max = grid_size_max
        log.info(
            f"{grid_size_max:>9d} {n_quads:>9d}"
            f" {size_imbalance:>12.0f}% {number_imbalance:>12.0f}%"
        )
    log.info(
        f"Selecting grid_size_max = {best_grid_size_max} for {n_processes} processes"
    )
    return best_grid_size_max


def subdivide(quad_set: QuadSet, grid_size_max: int) -> SubQuadSet:
    """Subdividide `quad_set` till all grid dimensions are below `grid_size_max`."""
    # Divide edges and propagate using adjacency:
    divisions: list[list[Optional[np.ndarray]]] = [
        [None, None] for _ in range(len(quad_set.quads))
    ]

    def propagate_division(i_quad: int, i_edge: int, division: np.ndarray) -> None:
        """Propagate division to all unknown dimensions using adjacency recursively."""
        # Set specified quad, edge:
        division_flipped = division[::-1]
        divisions[i_quad][i_edge % 2] = division if (i_edge < 2) else division_flipped
        # Propagate by adjacency (on both edges along this dimension):
        # Note that the division is flipped for the adjacent edge in the other quad
        for i_edge_equiv, division_equiv in (
            (i_edge, division_flipped),
            ((i_edge + 2) % 4, division),
        ):
            j_quad, j_edge = quad_set.adjacency[i_quad, i_edge_equiv]
            if j_quad >= 0 and divisions[j_quad][j_edge % 2] is None:
                propagate_division(j_quad, j_edge, division_equiv)

    for i_quad_cur, quad_divisions_i in enumerate(divisions):
        for i_edge_cur in range(2):
            if quad_divisions_i[i_edge_cur] is None:
                # Not yet divided: determine and propagate division:
                grid_size_cur = quad_set.grid_size[i_quad_cur, i_edge_cur]
                division_cur = split_evenly(grid_size_cur, grid_size_max)
                propagate_division(i_quad_cur, i_edge_cur, division_cur)

    # Divide quads
    quad_divisions = np.zeros((len(divisions), 2), dtype=int)
    for i_div, divisions_i in enumerate(divisions):
        for j_div, division_ij in enumerate(divisions_i):
            assert division_ij is not None
            quad_divisions[i_div, j_div] = len(division_ij)
    n_sub_quads = np.prod(quad_divisions, axis=1)
    n_sub_quads_prev = np.concatenate(([0], np.cumsum(n_sub_quads)))
    n_sub_quads_tot = n_sub_quads_prev[-1]
    quad_index = np.empty(n_sub_quads_tot, dtype=int)
    grid_start = np.empty((n_sub_quads_tot, 2), dtype=int)
    grid_stop = np.empty((n_sub_quads_tot, 2), dtype=int)
    adjacency = np.empty((n_sub_quads_tot, 4, 2), dtype=int)
    has_apertures = np.empty((n_sub_quads_tot, 4), dtype=bool)
    for i_quad, (div0, div1) in enumerate(divisions):
        cur_slice = slice(n_sub_quads_prev[i_quad], n_sub_quads_prev[i_quad + 1])
        quad_index[cur_slice] = i_quad
        # Grid ranges within quad
        assert div0 is not None
        assert div1 is not None
        grid_splits0 = np.concatenate((np.zeros(1, dtype=int), np.cumsum(div0)))
        grid_splits1 = np.concatenate((np.zeros(1, dtype=int), np.cumsum(div1)))
        grid_splits = np.stack(
            np.meshgrid(grid_splits0, grid_splits1, indexing="ij"), axis=2
        )
        grid_start[cur_slice] = grid_splits[:-1, :-1].reshape(-1, 2)
        grid_stop[cur_slice] = grid_splits[1:, 1:].reshape(-1, 2)

        # Adjacency within current quad:
        adjacency_slice = np.full((len(div0), len(div1), 4, 2), -1)
        has_apertures_slice = np.full(adjacency_slice.shape[:-1], False)
        i_sub_quads = np.arange(cur_slice.start, cur_slice.stop).reshape(
            *quad_divisions[i_quad]
        )
        for i_edge, i_slice in enumerate(INTERIOR_SLICES):
            j_edge = (i_edge + 2) % 4  # 0-edges connect to 2-edges etc.
            j_slice_int = INTERIOR_SLICES[j_edge]
            adjacency_slice[i_slice + (i_edge, 0)] = i_sub_quads[j_slice_int]
            adjacency_slice[i_slice + (i_edge, 1)] = j_edge

        # Adjacency between quads:
        for i_edge, (j_quad, j_edge) in enumerate(quad_set.adjacency[i_quad]):
            i_slice_edge = BOUNDARY_SLICES[i_edge] + (i_edge,)
            if j_quad >= 0:
                j_slice_ext = BOUNDARY_SLICES[j_edge]
                j_sub_quads = np.arange(
                    n_sub_quads_prev[j_quad], n_sub_quads_prev[j_quad + 1]
                ).reshape(*quad_divisions[j_quad])
                adjacency_slice[i_slice_edge + (0,)] = j_sub_quads[j_slice_ext][::-1]
                adjacency_slice[i_slice_edge + (1,)] = j_edge
            has_apertures_slice[i_slice_edge] = quad_set.has_apertures[i_quad, i_edge]
        adjacency[cur_slice] = adjacency_slice.reshape(-1, 4, 2)
        has_apertures[cur_slice] = has_apertures_slice.reshape(-1, 4)

    return SubQuadSet(quad_index, grid_start, grid_stop, adjacency, has_apertures)


def split_evenly(n_tasks: int, max_tasks: int) -> np.ndarray:
    """Split `n_tasks` as close to evenly with `max_tasks` per bins."""
    n_split = ceildiv(n_tasks, max_tasks)
    split_points = (np.arange(n_split + 1) * n_tasks) // n_split
    return np.diff(split_points)


# Slice all but boundary for each edge orientation
# This is always used in known combinations where reversed slicing
# is factored out analytically (2- and 3-edges are traversed oppositely)
INTERIOR_SLICES = (
    (slice(None), slice(1, None)),
    (slice(0, -1), slice(None)),
    (slice(None), slice(0, -1)),
    (slice(1, None), slice(None)),
)


# Slice boundary only for each edge orientation
# This accounts for the correct order of traversal of 2- and 3-edges,
# as it is used in arbitrary adjacency cases
BOUNDARY_SLICES = (
    (slice(None), 0),
    (-1, slice(None)),
    (slice(-1, None, -1), -1),
    (0, slice(-1, None, -1)),
)
