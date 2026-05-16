"""Precomputed per-target reference observables, cached to disk.

`load_reference_observables(chain)` returns everything `premval.scoring`
recomputes on every call today: the CA atom indices, the CA-only superposed
reference xyz, the crystal CA frame, the PCA fit on the reference, and the
per-atom moments + contact probability matrix on AlphaFlow's 1000-frame
subsample (seed 137). First call computes + saves; later calls memo-load
from `~/.cache/premval/references/{kind}/{chain}.npz`.

The scorer does NOT consume this cache yet (deliberate; integration is a
separate change). The dataclass / disk layout is the API we'll wire in.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from premval.data.atlas import AtlasKind, default_cache_dir, load_chain_trajectory
from premval.metrics.alphaflow_port import get_mean_covar
from premval.metrics.panel import ALPHAFLOW_SEED, CONTACT_THRESHOLD_NM

if TYPE_CHECKING:
    import mdtraj as md


@dataclasses.dataclass(frozen=True)
class ReferenceObservables:
    ca_indices: NDArray[np.int64]
    ref_xyz_ca: NDArray[np.float32]
    crystal_xyz_ca: NDArray[np.float32]
    pca_components: NDArray[np.float32]
    pca_mean: NDArray[np.float32]
    pca_explained_variance: NDArray[np.float32]
    ref_mean: NDArray[np.float32]  # (n_atoms, 3), nm; per-atom mean over subsample
    ref_covar: NDArray[np.float32]  # (n_atoms, 3, 3), nm^2; per-atom covar (rmwd input)
    ref_rmsf: NDArray[np.float32]  # (n_atoms,), A; per-atom RMSF vs crystal frame
    ref_contact_prob: NDArray[np.float32]


_N_PCA_COMPONENTS = 50
_SUBSAMPLE_N = 1000


def cache_path(chain: str, kind: str, cache_dir: Path) -> Path:
    return cache_dir / "references" / kind / f"{chain}.npz"


def _ca_indices(top: md.Topology) -> NDArray[np.int64]:
    return np.array([a.index for a in top.atoms if a.name == "CA"], dtype=np.int64)


def _subsample(arr: NDArray[np.floating], n: int, seed: int) -> NDArray[np.floating]:
    if n >= arr.shape[0]:
        return arr
    rng = np.random.default_rng(seed)
    return arr[np.sort(rng.choice(arr.shape[0], size=n, replace=False))]


def _compute(chain: str, kind: AtlasKind, cache_dir: Path) -> ReferenceObservables:
    import mdtraj

    traj = load_chain_trajectory(chain, kind=kind, cache_dir=cache_dir)
    ca_idx = _ca_indices(traj.topology)
    traj_ca = traj.atom_slice(ca_idx)
    traj_ca.superpose(traj_ca, frame=0)

    ref_xyz = traj_ca.xyz.astype(np.float32)
    crystal_xyz = ref_xyz[0].copy()

    n_frames, n_res, _ = ref_xyz.shape
    flat = ref_xyz.reshape(n_frames, n_res * 3)
    n_components = min(_N_PCA_COMPONENTS, n_frames, n_res * 3)
    pca = PCA(n_components=n_components)
    pca.fit(flat)

    sub_xyz = _subsample(ref_xyz, _SUBSAMPLE_N, ALPHAFLOW_SEED)
    ref_mean, ref_covar = get_mean_covar(sub_xyz)

    distmat = np.linalg.norm(sub_xyz[:, None, :] - sub_xyz[:, :, None], axis=-1)
    ref_contact = (distmat < CONTACT_THRESHOLD_NM).mean(0).astype(np.float32)

    # Match panel.rmsf_correlation: ×10 to convert nm → A.
    ref_rmsf = mdtraj.rmsf(traj_ca, traj_ca[0]) * 10

    return ReferenceObservables(
        ca_indices=ca_idx,
        ref_xyz_ca=ref_xyz,
        crystal_xyz_ca=crystal_xyz,
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        ref_mean=ref_mean.astype(np.float32),
        ref_covar=ref_covar.astype(np.float32),
        ref_rmsf=ref_rmsf.astype(np.float32),
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
        ref_rmsf=obs.ref_rmsf,
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
        ref_rmsf=data["ref_rmsf"],
        ref_contact_prob=data["ref_contact_prob"],
    )


def load_reference_observables(
    chain: str,
    kind: AtlasKind = "analysis",
    cache_dir: Path | None = None,
) -> ReferenceObservables:
    """Load cached reference observables for `chain`; compute and save if missing.

    Cache layout: `{cache_dir}/references/{kind}/{chain}.npz`. Default
    `cache_dir` is `default_cache_dir()` (i.e. `~/.cache/premval`).
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    path = cache_path(chain, kind, cache_dir)
    if path.exists():
        return _load_from_disk(path)
    obs = _compute(chain, kind, cache_dir)
    _save(obs, path)
    return obs
