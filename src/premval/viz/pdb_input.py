"""Load a list of single-structure PDB files into one `mdtraj.Trajectory`.

Generative ensemble models typically emit a bag of single-frame PDB files
rather than an XTC trajectory. This module is the bridge: it loads each
file, validates that every PDB carries an identical topology, and joins
them into a multi-frame `Trajectory` (one frame per input PDB).

The strict topology check is intentional. `mdtraj.join` will silently
concatenate xyz arrays even if atom orders differ, producing a trajectory
where "atom 5" means different things in different frames. We fail fast
instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import mdtraj as md

if TYPE_CHECKING:
    pass


def load_pdb_set(paths: Sequence[Path | str]) -> md.Trajectory:
    """Load single-frame PDBs and join them into one trajectory.

    Args:
        paths: Ordered sequence of PDB file paths. Each file must contain
            exactly one model and share an identical topology with the
            first file in the sequence.

    Returns:
        A `mdtraj.Trajectory` with `n_frames == len(paths)`, frames in
        input order, and the topology of the first input.

    Raises:
        ValueError: If `paths` is empty, any PDB has more than one frame,
            or any topology differs from the first.
    """
    if not paths:
        raise ValueError("load_pdb_set requires at least one PDB path")

    reference: md.Trajectory | None = None
    frames: list[md.Trajectory] = []
    for path in paths:
        traj = md.load(str(path))
        if traj.n_frames != 1:
            raise ValueError(
                f"{path}: expected 1 frame, got {traj.n_frames}. "
                "load_pdb_set is for single-structure PDBs; "
                "use md.load directly for multi-model files."
            )
        if reference is None:
            reference = traj
        else:
            _require_matching_topology(reference, traj, path)
        frames.append(traj)

    return md.join(frames, check_topology=True)


def _require_matching_topology(
    reference: md.Trajectory, candidate: md.Trajectory, candidate_path: Path | str
) -> None:
    ref_top = reference.topology
    cand_top = candidate.topology
    if ref_top.n_atoms != cand_top.n_atoms:
        raise ValueError(
            f"{candidate_path}: topology mismatch "
            f"(reference has {ref_top.n_atoms} atoms, this PDB has {cand_top.n_atoms})"
        )
    ref_atoms = [(a.name, a.residue.name, a.residue.index) for a in ref_top.atoms]
    cand_atoms = [(a.name, a.residue.name, a.residue.index) for a in cand_top.atoms]
    if ref_atoms != cand_atoms:
        first_diff = next(
            (i for i, (r, c) in enumerate(zip(ref_atoms, cand_atoms, strict=True)) if r != c),
            None,
        )
        raise ValueError(
            f"{candidate_path}: topology differs from reference at atom index {first_diff} "
            f"(reference={ref_atoms[first_diff] if first_diff is not None else None}, "
            f"this={cand_atoms[first_diff] if first_diff is not None else None})"
        )
