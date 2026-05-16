"""End-to-end scoring: submission ensemble vs ATLAS reference, one target.

`score(submission, reference)` is the single function the CLI calls; it
aligns CA atoms by residue index, superposes onto the reference's first
frame, and runs the four-metric panel. `score_chain(submission_path,
chain)` is the path-friendly convenience wrapper.

The v1 panel is CA-only across the board so there is a single alignment
path (matches AlphaFlow's `--ca_only` mode). Heavy-atom RMSF is an obvious
fidelity upgrade once the leaderboard is live, but it requires a second
topology-alignment pass and we deliberately defer that complexity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from premval.data import load_chain_trajectory
from premval.io import enforce_ensemble_size, load_ensemble
from premval.metrics.panel import (
    contact_jaccard,
    md_pca_w2,
    rmsf_correlation,
    rmwd,
)
from premval.topology import select_matched_ca

if TYPE_CHECKING:
    import mdtraj as md

REFERENCE_SUBSAMPLE = 1000


def score(
    submission: md.Trajectory,
    reference: md.Trajectory,
) -> dict[str, Any]:
    """Compute the v1 metric panel for one (submission, reference) pair.

    Both trajectories are sliced to their matched CA atoms (by
    `(chain.index, resSeq)`), superposed onto the reference's first frame,
    and fed through the four panel metrics. The reference is subsampled to
    `REFERENCE_SUBSAMPLE` frames inside the moment-based metrics (RMWD,
    contacts) so per-target compute stays bounded for long MD runs.

    Args:
        submission: Submission ensemble (any frame count, any atom set
            that overlaps with the reference at the CA level).
        reference: Reference ensemble (e.g. ATLAS MD concat). The first
            frame is used as the crystal/origin for RMSF and contacts.

    Returns:
        Dict with keys `n_residues`, `n_ref_frames`, `n_sub_frames`, plus
        the four panel results (`rmsf_pearson`, `rmwd`, `md_pca_w2`,
        `weak_contacts_jaccard`, `transient_contacts_jaccard`) and the
        raw RMWD components (`emd_mean_rms`, `emd_var_rms`).
    """
    ref_ca_idx, sub_ca_idx = select_matched_ca(reference, submission)
    ref_ca = reference.atom_slice(ref_ca_idx)
    sub_ca = submission.atom_slice(sub_ca_idx)
    crystal_ca = ref_ca[0]

    ref_ca = ref_ca.superpose(crystal_ca)
    sub_ca = sub_ca.superpose(crystal_ca)

    rmsf_out = rmsf_correlation(ref_ca, sub_ca, crystal_ca)
    rmwd_out = rmwd(ref_ca.xyz, sub_ca.xyz, reference_subsample_size=REFERENCE_SUBSAMPLE)
    md_pca = md_pca_w2(ref_ca.xyz, sub_ca.xyz)
    contacts = contact_jaccard(
        ref_ca.xyz,
        sub_ca.xyz,
        crystal_ca.xyz[0],
        reference_subsample_size=REFERENCE_SUBSAMPLE,
    )

    return {
        "n_residues": int(ref_ca.n_atoms),
        "n_ref_frames": int(ref_ca.n_frames),
        "n_sub_frames": int(sub_ca.n_frames),
        "rmsf_pearson": float(rmsf_out["rmsf_pearson"]),
        "emd_mean_rms": rmwd_out["emd_mean_rms"],
        "emd_var_rms": rmwd_out["emd_var_rms"],
        "rmwd": rmwd_out["rmwd"],
        "md_pca_w2": float(md_pca),
        "weak_contacts_jaccard": contacts["weak_contacts_jaccard"],
        "transient_contacts_jaccard": contacts["transient_contacts_jaccard"],
    }


def score_chain(
    submission_path: Path | str,
    chain: str,
    *,
    enforce_size: int | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Convenience: load a submission PDB and ATLAS chain, then score.

    Args:
        submission_path: Path to a multi-model submission PDB.
        chain: ATLAS chain identifier (e.g. `6cka_B`); must be cached
            locally (run `premval fetch` first).
        enforce_size: If set, require the submission to have exactly this
            many frames (subsample if larger, reject if smaller).
        cache_dir: ATLAS cache root. Defaults to `default_cache_dir()`.

    Returns:
        The same dict as `score`, plus `chain` and `submission_path`.
    """
    submission = load_ensemble(submission_path)
    if enforce_size is not None:
        submission = enforce_ensemble_size(submission, expected=enforce_size)
    reference = load_chain_trajectory(chain, cache_dir=cache_dir)
    result = score(submission, reference)
    result["chain"] = chain
    result["submission_path"] = str(submission_path)
    return result
