from __future__ import annotations
import argparse

from qimpy import rc
from . import SVGParser, plot_spline


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()

    # Find each patch and print its respective vertices
    patch_set = SVGParser(args.input_svg).patch_set

    print(f"Found {len(patch_set.quads)} quad-patches:")
    for i_quad, (quad, adjacency) in enumerate(
        zip(patch_set.quads, patch_set.adjacency)
    ):
        print(f"Quad {i_quad}:")
        for i_edge, edge_index in enumerate(quad):
            i_verts = patch_set.edges[edge_index, [0, -1]]
            v_start, v_stop = patch_set.vertices[i_verts].to(rc.cpu).numpy()
            neigh_patch, neigh_edge = adjacency[i_edge].to(rc.cpu).numpy()
            if neigh_patch >= 0:
                neigh_str = f" (Neighbor: quad {neigh_patch} edge {neigh_edge})"
            else:
                neigh_str = ""
            print(f"  Edge {i_edge}: {v_start} -> {v_stop}{neigh_str}")

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for edge in patch_set.edges:
        plot_spline(ax, patch_set.vertices[edge])
    plt.show()


if __name__ == "__main__":
    main()
