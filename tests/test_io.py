from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from premval.io import enforce_ensemble_size, load_ensemble


def _toy_traj(n_frames: int, n_residues: int = 5, seed: int = 0) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("CA", md.element.carbon, res)
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_frames, n_residues, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def test_load_ensemble_reads_multimodel_pdb(tmp_path: Path) -> None:
    traj = _toy_traj(n_frames=4)
    path = tmp_path / "ensemble.pdb"
    traj.save_pdb(str(path))
    loaded = load_ensemble(path)
    assert loaded.n_frames == 4
    assert loaded.topology.n_atoms == traj.topology.n_atoms


def test_load_ensemble_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ensemble(tmp_path / "nope.pdb")


def test_enforce_ensemble_size_passes_through_exact(tmp_path: Path) -> None:
    traj = _toy_traj(n_frames=10)
    out = enforce_ensemble_size(traj, expected=10)
    assert out is traj


def test_enforce_ensemble_size_rejects_short(tmp_path: Path) -> None:
    traj = _toy_traj(n_frames=5)
    with pytest.raises(ValueError, match="requires 10"):
        enforce_ensemble_size(traj, expected=10)


def test_enforce_ensemble_size_subsamples_long_deterministically() -> None:
    traj = _toy_traj(n_frames=100)
    out_a = enforce_ensemble_size(traj, expected=25, seed=7)
    out_b = enforce_ensemble_size(traj, expected=25, seed=7)
    out_c = enforce_ensemble_size(traj, expected=25, seed=8)
    assert out_a.n_frames == 25
    np.testing.assert_array_equal(out_a.xyz, out_b.xyz)
    # Different seed should (with overwhelming probability) pick a different subset.
    assert not np.array_equal(out_a.xyz, out_c.xyz)
