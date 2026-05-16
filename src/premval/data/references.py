"""Precomputed per-target reference observables, cached to disk."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from premval.data import load_chain_trajectory
from premval.data.atlas import default_cache_dir
from premval.metrics.panel import (
    ALPHAFLOW_SEED,
    compute_contact_prob,
    compute_per_atom_stats,
    subsample,
)
from premval.topology import select_ca_indices, strip_hydrogens


@dataclasses.dataclass(frozen=True)
class ReferenceObservables:
    ca_indices: np.ndarray  # (n_residues,) int64 — indices into full-atom topology
    ref_xyz_ca: np.ndarray  # (n_ref_frames, n_residues, 3) float32 nm, superposed
    crystal_xyz_ca: np.ndarray  # (n_residues, 3) float32 nm — first-frame CA coords
    pca_components: np.ndarray  # (n_components, n_residues*3) float32
    pca_mean: np.ndarray  # (n_residues*3,) float32
    pca_explained_variance: np.ndarray  # (n_components,) float32
    ref_mean: np.ndarray  # (n_residues, 3) float32 — per-atom mean over subsample
    ref_covar: np.ndarray  # (n_residues*3, n_residues*3) float32
    ref_contact_prob: np.ndarray  # (n_residues, n_residues) float32


_N_PCA_COMPONENTS = 50
_CONTACT_THRESHOLD_NM = 0.8
_SUBSAMPLE_N = 1000


def _cache_path(chain: str, kind: str, cache_dir: Path) -> Path:
    return cache_dir / "references" / kind / f"{chain}.npz"


def _compute(chain: str, kind: str, cache_dir: Path) -> ReferenceObservables:
    import mdtraj

    traj: mdtraj.Trajectory = load_chain_trajectory(chain, kind=kind, cache_dir=cache_dir)
    traj = strip_hydrogens(traj)
    ca_idx = select_ca_indices(traj.topology)
    traj_ca: mdtraj.Trajectory = traj.atom_slice(ca_idx)
    traj_ca.superpose(traj_ca, frame=0)

    ref_xyz = traj_ca.xyz.astype(np.float32)  # (n_frames, n_res, 3)
    crystal_xyz = ref_xyz[0].copy()  # (n_res, 3)

    n_frames, n_res, _ = ref_xyz.shape
    flat = ref_xyz.reshape(n_frames, n_res * 3)
    n_components = min(_N_PCA_COMPONENTS, n_frames, n_res * 3)
    pca: PCA = PCA(n_components=n_components)
    pca.fit(flat)

    sub_xyz = subsample(ref_xyz, n=_SUBSAMPLE_N, seed=ALPHAFLOW_SEED)
    ref_mean, ref_covar = compute_per_atom_stats(sub_xyz)
    ref_contact = compute_contact_prob(sub_xyz, threshold_nm=_CONTACT_THRESHOLD_NM)

    return ReferenceObservables(
        ca_indices=ca_idx.astype(np.int64),
        ref_xyz_ca=ref_xyz,
        crystal_xyz_ca=crystal_xyz,
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        ref_mean=ref_mean,
        ref_covar=ref_covar,
        ref_contact_prob=ref_contact,
    )


def _save(obs: ReferenceObservables, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ca_indices=obs.ca_indices,
        ref_xyz_ca=obs.ref_xyz_ca,
        crystal_xyz_ca=obs.crystal_xyz_ca,
        pca_components=obs.pca_components,
        pca_mean=obs.pca_mean,
        pca_explained_variance=obs.pca_explained_variance,
        ref_mean=obs.ref_mean,
        ref_covar=obs.ref_covar,
        ref_contact_prob=obs.ref_contact_prob,
    )


def _load_from_disk(path: Path) -> ReferenceObservables:
    data = np.load(path)
    return ReferenceObservables(
        ca_indices=data["ca_indices"],
        ref_xyz_ca=data["ref_xyz_ca"],
        crystal_xyz_ca=data["crystal_xyz_ca"],
        pca_components=data["pca_components"],
        pca_mean=data["pca_mean"],
        pca_explained_variance=data["pca_explained_variance"],
        ref_mean=data["ref_mean"],
        ref_covar=data["ref_covar"],
        ref_contact_prob=data["ref_contact_prob"],
    )


def load_reference_observables(
    chain: str,
    kind: str = "analysis",
    cache_dir: Path | None = None,
) -> ReferenceObservables:
    """
    Load cached reference observables for chain, computing and saving if missing.

    Cache location: {cache_dir}/references/{kind}/{chain}.npz
    Default cache_dir: ~/.cache/premval
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    path = _cache_path(chain, kind, cache_dir)
    if path.exists():
        return _load_from_disk(path)
    obs = _compute(chain, kind, cache_dir)
    _save(obs, path)
    return obs
