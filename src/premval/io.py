"""Load and normalize ensemble submissions.

A v1 submission is a multi-model PDB file: one MODEL block per sampled
conformation. AlphaFlow/ESMFlow emit exactly this (`protein.prots_to_pdb`),
and PED IDP ensembles use the same format.

The leaderboard contract is **250 frames per target**; the AlphaFlow paper
is explicit that ensemble metrics are not comparable across sample counts.
`enforce_ensemble_size` is the choke-point that enforces this: it rejects
short submissions outright and deterministically subsamples long ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mdtraj as md

DEFAULT_ENSEMBLE_SIZE = 250


def load_ensemble(pdb_path: Path | str) -> md.Trajectory:
    """Load a multi-model PDB as a single trajectory.

    Args:
        pdb_path: Path to a PDB file with one or more MODEL blocks.

    Returns:
        An `mdtraj.Trajectory` with `n_frames == number of MODEL blocks`.

    Raises:
        FileNotFoundError: If `pdb_path` does not exist.
        ValueError: If the file loads but contains zero frames.
    """
    import mdtraj as md

    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"Ensemble PDB not found: {path}")
    traj = md.load(str(path))
    if traj.n_frames == 0:
        raise ValueError(f"{path}: PDB loaded with zero frames")
    return traj


def enforce_ensemble_size(
    traj: md.Trajectory,
    *,
    expected: int = DEFAULT_ENSEMBLE_SIZE,
    seed: int = 0,
) -> md.Trajectory:
    """Return a trajectory with exactly `expected` frames.

    Submissions shorter than `expected` are rejected (the AlphaFlow paper
    notes that metrics are sample-count dependent, so silently scoring a
    100-frame ensemble against a 250-frame contract would corrupt the
    leaderboard). Submissions longer than `expected` are deterministically
    subsampled with `seed`, giving reproducible scores across re-runs.

    Args:
        traj: Input trajectory.
        expected: Required frame count. Defaults to 250.
        seed: RNG seed for the subsample. Same seed + same input gives the
            same selection across runs and across machines.

    Returns:
        The original trajectory if `traj.n_frames == expected`, otherwise
        a frame-subset view.

    Raises:
        ValueError: If `traj.n_frames < expected`.
    """
    n = traj.n_frames
    if n < expected:
        raise ValueError(
            f"Ensemble has {n} frames; the leaderboard requires {expected}. "
            "AlphaFlow's evaluation is not comparable across sample counts."
        )
    if n == expected:
        return traj
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=expected, replace=False))
    return traj[indices]
