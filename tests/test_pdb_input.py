from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from premval.viz.pdb_input import load_pdb_set


def _toy_single_frame_traj(n_residues: int = 5, seed: int = 0) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)
    rng = np.random.default_rng(seed)
    xyz = rng.normal(scale=0.5, size=(1, n_residues * 4, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def _write_frames(tmp_path: Path, count: int, seeds: list[int]) -> list[Path]:
    paths: list[Path] = []
    for i, seed in enumerate(seeds[:count]):
        traj = _toy_single_frame_traj(seed=seed)
        path = tmp_path / f"frame_{i:03d}.pdb"
        traj.save_pdb(str(path))
        paths.append(path)
    return paths


def test_load_pdb_set_joins_into_one_trajectory(tmp_path: Path) -> None:
    paths = _write_frames(tmp_path, 4, [0, 1, 2, 3])
    traj = load_pdb_set(paths)
    assert traj.n_frames == 4
    assert traj.topology.n_atoms == 5 * 4


def test_load_pdb_set_accepts_str_and_path(tmp_path: Path) -> None:
    paths = _write_frames(tmp_path, 2, [0, 1])
    traj = load_pdb_set([str(paths[0]), paths[1]])
    assert traj.n_frames == 2


def test_load_pdb_set_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one PDB"):
        load_pdb_set([])


def test_load_pdb_set_rejects_multi_frame(tmp_path: Path) -> None:
    top = _toy_single_frame_traj().topology
    xyz = np.zeros((3, top.n_atoms, 3), dtype=np.float32)
    multi = md.Trajectory(xyz, top)
    path = tmp_path / "multi.pdb"
    multi.save_pdb(str(path))
    with pytest.raises(ValueError, match="expected 1 frame, got 3"):
        load_pdb_set([path])


def test_load_pdb_set_rejects_topology_mismatch(tmp_path: Path) -> None:
    small = _toy_single_frame_traj(n_residues=3)
    big = _toy_single_frame_traj(n_residues=5)
    p1 = tmp_path / "a.pdb"
    p2 = tmp_path / "b.pdb"
    small.save_pdb(str(p1))
    big.save_pdb(str(p2))
    with pytest.raises(ValueError, match="topology mismatch"):
        load_pdb_set([p1, p2])


def _toy_traj_with_residue(n_residues: int, residue_name: str) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue(residue_name, chain)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)
    xyz = np.zeros((1, n_residues * 4, 3), dtype=np.float32)
    return md.Trajectory(xyz, top)


def test_load_pdb_set_reports_first_diverging_atom(tmp_path: Path) -> None:
    ref = _toy_traj_with_residue(3, "ALA")
    mismatched = _toy_traj_with_residue(3, "GLY")
    p1 = tmp_path / "ala.pdb"
    p2 = tmp_path / "gly.pdb"
    ref.save_pdb(str(p1))
    mismatched.save_pdb(str(p2))
    with pytest.raises(ValueError, match="topology differs from reference at atom index"):
        load_pdb_set([p1, p2])
