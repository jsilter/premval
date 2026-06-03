"""Permute trajectory frames so that adjacent frames are as similar as possible.

A bag of PDBs has no canonical order; playing them in arbitrary sequence
looks jittery. `smooth_order` returns a frame permutation that makes a
video as visually continuous as possible by keeping adjacent-frame Cα
RMSDs small.

Two algorithms:

- ``"cheapest_insertion"`` (default, O(N^2-3)): build an open path by
  seeding with the two most-distant frames, then for each remaining
  frame find the insertion position (and frame) that adds the least
  total path length. Standard TSP heuristic; typically within ~25% of
  optimal for metric distances and scales to thousands of frames.
- ``"optimal"`` (O(N^3) typical, O(N^4) worst): Bar-Joseph's optimal
  leaf ordering on a hierarchical clustering of the distance matrix
  (``scipy.cluster.hierarchy.optimal_leaf_ordering``). Optimal for the
  dendrogram-flip objective but expensive past N ≈ a few hundred.
"""

from __future__ import annotations

from typing import Literal

import mdtraj as md
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

OrderingMethod = Literal["cheapest_insertion", "optimal"]


def smooth_order(
    traj: md.Trajectory,
    atom_selection: str = "name CA",
    method: OrderingMethod = "cheapest_insertion",
) -> list[int]:
    """Return frame indices ordered for smoothest visual playback.

    Args:
        traj: Multi-frame trajectory whose frames should be permuted.
        atom_selection: mdtraj selection string for atoms used in the
            RMSD distance. Defaults to Cα atoms.
        method: Ordering algorithm. ``"cheapest_insertion"`` (default)
            scales well past a few hundred frames. ``"optimal"`` returns
            the global optimum of the dendrogram leaf-ordering objective
            but is expensive at large N.

    Returns:
        A permutation of ``range(traj.n_frames)`` that minimises the sum
        of Cα RMSDs between adjacent positions when used as a frame
        order.

    Raises:
        ValueError: If `traj` has fewer than two frames,
            `atom_selection` matches no atoms, or `method` is unknown.
    """
    n = traj.n_frames
    if n < 2:
        raise ValueError(f"smooth_order needs at least 2 frames, got {n}")
    atom_indices = traj.topology.select(atom_selection)
    if len(atom_indices) == 0:
        raise ValueError(f"atom selection {atom_selection!r} matched no atoms")

    distances = _pairwise_rmsd(traj, atom_indices)
    if n == 2:
        return [0, 1]

    if method == "cheapest_insertion":
        return _cheapest_insertion_order(distances)
    if method == "optimal":
        return _optimal_leaf_order(distances)
    raise ValueError(f"unknown ordering method {method!r}")


def _pairwise_rmsd(traj: md.Trajectory, atom_indices: np.ndarray) -> np.ndarray:
    n = traj.n_frames
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        distances[i] = md.rmsd(traj, traj, frame=i, atom_indices=atom_indices)
    # md.rmsd's tiny asymmetries (1e-7 level) trip downstream strict-symmetry
    # checks (squareform, hierarchy); symmetrise explicitly.
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    return distances


def _optimal_leaf_order(distances: np.ndarray) -> list[int]:
    condensed = squareform(distances, checks=False)
    z = linkage(condensed, method="average")
    z_ordered = optimal_leaf_ordering(z, condensed)
    return [int(i) for i in leaves_list(z_ordered)]


def _cheapest_insertion_order(distances: np.ndarray) -> list[int]:
    """Open-path cheapest-insertion heuristic.

    Seeds the path with the two most-distant frames, then repeatedly
    selects the (frame, position) pair minimising the increase in total
    path length. The two seeds naturally end up as the endpoints, which
    matches the intuition that the most dissimilar frames should bracket
    the video.
    """
    n = distances.shape[0]
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    path: list[int] = [int(i), int(j)]
    unvisited = [k for k in range(n) if k != path[0] and k != path[1]]

    while unvisited:
        path_arr = np.asarray(path, dtype=np.intp)
        # Edge costs between consecutive path vertices: D[path[k-1], path[k]].
        edge_costs = distances[path_arr[:-1], path_arr[1:]]
        best_global_cost = np.inf
        best_f = -1
        best_pos = -1
        for f in unvisited:
            d_to_path = distances[f, path_arr]
            costs = np.empty(len(path) + 1, dtype=np.float64)
            costs[0] = d_to_path[0]
            costs[-1] = d_to_path[-1]
            if len(path) >= 2:
                costs[1:-1] = d_to_path[:-1] + d_to_path[1:] - edge_costs
            k_best = int(np.argmin(costs))
            if costs[k_best] < best_global_cost:
                best_global_cost = float(costs[k_best])
                best_f = f
                best_pos = k_best
        path.insert(best_pos, best_f)
        unvisited.remove(best_f)

    return path
