from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from premval.viz.ordering import OrderingMethod, smooth_order

_METHODS: list[OrderingMethod] = ["cheapest_insertion", "optimal"]


def _ca_only_trajectory(n_frames: int, n_residues: int = 10) -> md.Trajectory:
    """Build a trajectory whose frames are a linear interpolation between two
    random conformations. Frame `k` should be closest to frames `k-1` and `k+1`."""
    rng = np.random.default_rng(0)
    start = rng.normal(scale=2.0, size=(n_residues, 3)).astype(np.float32)
    end = rng.normal(scale=2.0, size=(n_residues, 3)).astype(np.float32)
    alphas = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    xyz = np.stack([(1 - a) * start + a * end for a in alphas])

    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("CA", md.element.carbon, res)
    return md.Trajectory(xyz, top)


@pytest.mark.parametrize("method", _METHODS)
def test_smooth_order_recovers_linear_interp_sequence(method: OrderingMethod) -> None:
    n_frames = 12
    traj = _ca_only_trajectory(n_frames)
    perm = np.array([7, 2, 11, 0, 5, 9, 3, 1, 10, 4, 8, 6])
    scrambled = traj[perm]

    order = smooth_order(scrambled, method=method)
    recovered_original_indices = perm[order]

    forward = list(range(n_frames))
    reverse = list(reversed(forward))
    assert recovered_original_indices.tolist() in (forward, reverse), (
        f"{method}: expected monotone traversal, got {recovered_original_indices.tolist()}"
    )


@pytest.mark.parametrize("method", _METHODS)
def test_smooth_order_returns_permutation_of_indices(method: OrderingMethod) -> None:
    traj = _ca_only_trajectory(6)
    order = smooth_order(traj, method=method)
    assert sorted(order) == list(range(6))


@pytest.mark.parametrize("method", _METHODS)
def test_smooth_order_handles_two_frames(method: OrderingMethod) -> None:
    traj = _ca_only_trajectory(2)
    assert smooth_order(traj, method=method) == [0, 1]


def test_smooth_order_rejects_single_frame() -> None:
    traj = _ca_only_trajectory(1)
    with pytest.raises(ValueError, match="at least 2 frames"):
        smooth_order(traj)


def test_smooth_order_rejects_empty_atom_selection() -> None:
    traj = _ca_only_trajectory(4)
    with pytest.raises(ValueError, match="matched no atoms"):
        smooth_order(traj, atom_selection="name DOESNOTEXIST")


def test_smooth_order_rejects_unknown_method() -> None:
    traj = _ca_only_trajectory(4)
    with pytest.raises(ValueError, match="unknown ordering method"):
        smooth_order(traj, method="nonsense")  # type: ignore[arg-type]


def test_cheapest_insertion_beats_random_total_length() -> None:
    """Smoke check that cheapest insertion is at least as good as the
    average random permutation. Not a tight bound, just a sanity floor."""
    n_frames = 20
    traj = _ca_only_trajectory(n_frames)

    rng = np.random.default_rng(42)
    random_perm = rng.permutation(n_frames)
    scrambled = traj[random_perm]

    order = smooth_order(scrambled, method="cheapest_insertion")
    ordered = scrambled[order]

    def total_path_length(t: md.Trajectory) -> float:
        ca = t.topology.select("name CA")
        return float(
            sum(md.rmsd(t, t, frame=k, atom_indices=ca)[k + 1] for k in range(t.n_frames - 1))
        )

    random_len = total_path_length(scrambled)
    ordered_len = total_path_length(ordered)
    assert ordered_len < random_len, (
        f"cheapest insertion did not improve over random: "
        f"random={random_len:.3f}, ordered={ordered_len:.3f}"
    )
