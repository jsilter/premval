"""Align a submission topology to a reference topology by CA atoms.

The reference (e.g. ATLAS MD) and a submission (e.g. AlphaFlow output) are
both all-atom but may disagree in residue presence or atom ordering. The
metric panel operates on CA traces (per AlphaFlow's `analyze_ensembles.py`),
so the alignment we need is "which CA atom in each topology corresponds to
the same residue."

Residues are matched by `(chain.index, residue.resSeq)` — `resSeq` is the
stable PDB sequence number (a residue number from the deposited structure),
while `residue.index` is just a position-in-file counter that drifts when
residues are inserted or removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mdtraj as md


def select_matched_ca(
    reference: md.Trajectory,
    submission: md.Trajectory,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CA atom indices in `reference` and `submission` matched by residue.

    Each residue is keyed by `(chain.index, residue.resSeq)`. The returned
    arrays are aligned: index `k` in both arrays refers to the same residue.
    Residues present in only one topology are dropped.

    Args:
        reference: Reference trajectory (e.g. ATLAS MD ensemble).
        submission: Submission trajectory (e.g. AlphaFlow output).

    Returns:
        `(ref_indices, sub_indices)`, two int arrays of equal length. Use
        them to slice CA coordinates: `reference.xyz[:, ref_indices, :]`
        and `submission.xyz[:, sub_indices, :]`.

    Raises:
        ValueError: If no CA residues match between the two topologies.
    """
    ref_ca = _ca_map(reference)
    sub_ca = _ca_map(submission)
    shared_keys = sorted(ref_ca.keys() & sub_ca.keys())
    if not shared_keys:
        raise ValueError(
            f"No matching CA residues between reference ({len(ref_ca)} CA) "
            f"and submission ({len(sub_ca)} CA)."
        )
    ref_indices = np.array([ref_ca[k] for k in shared_keys], dtype=np.int64)
    sub_indices = np.array([sub_ca[k] for k in shared_keys], dtype=np.int64)
    return ref_indices, sub_indices


def _ca_map(traj: md.Trajectory) -> dict[tuple[int, int], int]:
    """Build a `(chain_index, resSeq) -> atom_index` map for CA atoms."""
    result: dict[tuple[int, int], int] = {}
    for atom in traj.topology.atoms:
        if atom.name != "CA":
            continue
        key = (atom.residue.chain.index, atom.residue.resSeq)
        if key in result:
            # Duplicate residue keys break the alignment contract; surface it
            # rather than silently overwriting the earlier hit.
            raise ValueError(
                f"Duplicate CA for residue {key} in topology; cannot align unambiguously."
            )
        result[key] = atom.index
    return result
