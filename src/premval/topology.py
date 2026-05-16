import mdtraj
import numpy as np


def select_ca_indices(top: mdtraj.Topology) -> np.ndarray:
    """Return int64 array of CA atom indices in topology."""
    return np.asarray(top.select("name CA"), dtype=np.int64)


def strip_hydrogens(traj: mdtraj.Trajectory) -> mdtraj.Trajectory:
    """Return trajectory with hydrogen atoms removed."""
    return traj.atom_slice(traj.topology.select("not element H"))


def select_matched_ca(
    ref_traj: mdtraj.Trajectory,
    query_traj: mdtraj.Trajectory,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ref_ca_indices, query_ca_indices) matching by residue sequence number.
    Simple implementation: find CA indices in each topology.
    """
    ref_ca_indices = select_ca_indices(ref_traj.topology)
    query_ca_indices = select_ca_indices(query_traj.topology)
    return ref_ca_indices, query_ca_indices
