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

from premval.data.atlas import default_cache_dir, load_chain_trajectory

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


def _compute(chain: str, kind: str, cache_dir: Path) -> ReferenceObservables:
    traj = load_chain_trajectory(chain, kind=kind, cache_dir=cache_dir)
    return compute_observables_from_traj(traj)


def compute_observables_from_traj(traj: md.Trajectory) -> ReferenceObservables:
    """Compute the full observables panel from an in-memory trajectory.

    Shared by the ATLAS reference path (`load_reference_observables`) and the
    model-samples path (`premval.data.samples.load_sample_observables`): both
    feed a CA-sliced, frame-0-superposed trajectory through the same PCA /
    moments / contact-probability / RMSF computation, so reference and sample
    overlays are computed identically and are directly comparable.

    Args:
        traj: A loaded trajectory (any number of frames; CA atoms are sliced
            out internally). Frame 0 is treated as the crystal/reference frame.

    Returns:
        The computed `ReferenceObservables`.
    """
    import mdtraj
    from sklearn.decomposition import PCA

    from premval.metrics.alphaflow_port import get_mean_covar
    from premval.metrics.panel import ALPHAFLOW_SEED, contact_probability

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

    ref_contact = contact_probability(sub_xyz).astype(np.float32)

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


def kabsch_matrix(
    mobile: NDArray[np.floating], target: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Return the 4x4 rigid transform mapping `mobile` onto `target` (min RMSD).

    Standard Kabsch superposition (rotation + translation, no scaling,
    reflection-corrected via the determinant sign). Both inputs are `(N, 3)`
    point sets in the same units; the returned homogeneous matrix is in those
    units and maps a column point `p` as `R @ p + t`.

    Args:
        mobile: Points to move, shape `(N, 3)`.
        target: Points to align onto, shape `(N, 3)`.

    Returns:
        A `(4, 4)` homogeneous transform.
    """
    mc = mobile.mean(axis=0)
    tc = target.mean(axis=0)
    h = (mobile - mc).T @ (target - tc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot
    matrix[:3, 3] = tc - rot @ mc
    return matrix


def save_observables(obs: ReferenceObservables, path: Path) -> None:
    """Write observables to `path` as a `.npz` (creating parent dirs)."""
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


def load_observables(path: Path) -> ReferenceObservables:
    """Load observables previously written by `save_observables`."""
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
    kind: str = "analysis",
    cache_dir: Path | None = None,
    *,
    force: bool = False,
) -> ReferenceObservables:
    """Load cached reference observables for `chain`; compute and save if missing.

    Cache layout: `{cache_dir}/references/{kind}/{chain}.npz`. Default
    `cache_dir` is `default_cache_dir()` (i.e. `~/.cache/premval`).

    Args:
        chain: PDB chain identifier such as `6cka_B`.
        kind: Cache namespace (ATLAS tier such as `analysis`, or another
            dataset name like `nanobody`); must match the cached bundle.
        cache_dir: Root cache directory.
        force: If True, recompute and overwrite the cache even when a
            `.npz` is already on disk. Use when the upstream code that
            produces references has changed.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    path = cache_path(chain, kind, cache_dir)
    if path.exists() and not force:
        return load_observables(path)
    obs = _compute(chain, kind, cache_dir)
    save_observables(obs, path)
    return obs
