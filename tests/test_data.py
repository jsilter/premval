"""Synthetic mdtraj.Trajectory helpers for testing."""
import mdtraj
import numpy as np


def make_synthetic_trajectory(
    n_frames: int = 50,
    n_residues: int = 10,
    seed: int = 42,
) -> mdtraj.Trajectory:
    """
    Create a synthetic CA-only mdtraj.Trajectory with random coordinates.

    Useful for testing without real ATLAS data.
    """
    rng = np.random.default_rng(seed)
    # Build a simple CA-only topology
    top = mdtraj.Topology()
    chain = top.add_chain()
    for i in range(n_residues):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", mdtraj.element.carbon, res)
    xyz = rng.standard_normal((n_frames, n_residues, 3)).astype(np.float32) * 0.5
    return mdtraj.Trajectory(xyz, top)


def make_full_atom_trajectory(
    n_frames: int = 50,
    n_residues: int = 10,
    seed: int = 42,
) -> mdtraj.Trajectory:
    """
    Create a synthetic full-atom trajectory (N, CA, C, O per residue) for testing.
    strip_hydrogens and select_ca_indices work on this.
    """
    rng = np.random.default_rng(seed)
    top = mdtraj.Topology()
    chain = top.add_chain()
    atom_names = ["N", "CA", "C", "O"]
    for i in range(n_residues):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        for name in atom_names:
            top.add_atom(name, mdtraj.element.carbon, res)
    n_atoms = n_residues * len(atom_names)
    xyz = rng.standard_normal((n_frames, n_atoms, 3)).astype(np.float32) * 0.5
    return mdtraj.Trajectory(xyz, top)
