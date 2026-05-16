from pathlib import Path

import mdtraj
import numpy as np
from sklearn.decomposition import PCA

from premval.data import load_chain_trajectory
from premval.metrics.panel import (
    ALPHAFLOW_SEED,
    compute_contact_prob,
    compute_per_atom_stats,
    contact_jaccard,
    md_pca_w2,
    rmwd,
    subsample,
)
from premval.topology import select_ca_indices, strip_hydrogens


def score(
    chain: str,
    submission_xyz: np.ndarray,  # (n_frames, n_residues, 3) CA-only, pre-aligned, nm
    kind: str = "analysis",
    cache_dir: Path | None = None,
) -> dict[str, float]:
    """
    Score a submission ensemble against the ATLAS reference for the given chain.

    Recomputes all reference observables on every call (expensive).
    """
    ref_traj: mdtraj.Trajectory = load_chain_trajectory(chain, kind=kind, cache_dir=cache_dir)
    ref_traj = strip_hydrogens(ref_traj)
    ca_idx = select_ca_indices(ref_traj.topology)
    ref_ca: mdtraj.Trajectory = ref_traj.atom_slice(ca_idx)

    # Superpose onto first frame (crystal)
    ref_ca.superpose(ref_ca, frame=0)
    ref_xyz = ref_ca.xyz.astype(np.float32)  # (n_ref_frames, n_res, 3)

    # PCA on reference
    n_ref, n_res, _ = ref_xyz.shape
    flat = ref_xyz.reshape(n_ref, n_res * 3)
    pca: PCA = PCA(n_components=min(50, n_ref, n_res * 3))
    pca.fit(flat)

    # 1000-frame subsample for stats
    sub_xyz = subsample(ref_xyz, n=1000, seed=ALPHAFLOW_SEED)
    ref_mean, ref_covar = compute_per_atom_stats(sub_xyz)
    ref_contact = compute_contact_prob(sub_xyz)

    # Score submission
    sub_mean, sub_covar = compute_per_atom_stats(submission_xyz)

    return {
        "rmwd": rmwd(sub_mean, sub_covar, ref_mean, ref_covar),
        "md_pca_w2": md_pca_w2(
            submission_xyz, pca.components_, pca.mean_, pca.explained_variance_
        ),
        "contact_jaccard": contact_jaccard(
            compute_contact_prob(submission_xyz),
            ref_contact,
        ),
    }
