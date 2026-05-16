from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from premval.topology import select_matched_ca


def _ca_traj(resseqs: list[int], n_frames: int = 1, seed: int = 0) -> md.Trajectory:
    """Build a CA-only trajectory with the given residue sequence numbers."""
    top = md.Topology()
    chain = top.add_chain()
    for resseq in resseqs:
        res = top.add_residue("ALA", chain, resSeq=resseq)
        top.add_atom("CA", md.element.carbon, res)
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_frames, len(resseqs), 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def test_select_matched_ca_identical_topologies() -> None:
    ref = _ca_traj([10, 11, 12, 13])
    sub = _ca_traj([10, 11, 12, 13])
    ref_idx, sub_idx = select_matched_ca(ref, sub)
    np.testing.assert_array_equal(ref_idx, [0, 1, 2, 3])
    np.testing.assert_array_equal(sub_idx, [0, 1, 2, 3])


def test_select_matched_ca_drops_unshared_residues() -> None:
    ref = _ca_traj([10, 11, 12, 13, 14])
    sub = _ca_traj([11, 12, 99])  # 99 not in ref, 10/13/14 not in sub
    ref_idx, sub_idx = select_matched_ca(ref, sub)
    # Shared residues = {11, 12}; ref has them at indices 1,2; sub at 0,1.
    np.testing.assert_array_equal(ref_idx, [1, 2])
    np.testing.assert_array_equal(sub_idx, [0, 1])


def test_select_matched_ca_no_overlap_raises() -> None:
    ref = _ca_traj([1, 2, 3])
    sub = _ca_traj([100, 101])
    with pytest.raises(ValueError, match="No matching CA residues"):
        select_matched_ca(ref, sub)


def test_select_matched_ca_ignores_non_ca_atoms() -> None:
    """An all-atom topology should yield the same CA mapping as a CA-only one."""
    top = md.Topology()
    chain = top.add_chain()
    for resseq in (5, 6, 7):
        res = top.add_residue("ALA", chain, resSeq=resseq)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
    xyz = np.zeros((1, 9, 3), dtype=np.float32)
    all_atom = md.Trajectory(xyz, top)

    ca_only = _ca_traj([5, 6, 7])
    ref_idx, sub_idx = select_matched_ca(all_atom, ca_only)
    # CA atoms in all_atom are at positions 1, 4, 7.
    np.testing.assert_array_equal(ref_idx, [1, 4, 7])
    np.testing.assert_array_equal(sub_idx, [0, 1, 2])
