"""The v1 PREMVAL metric panel, ported from AlphaFlow.

Functions in this module take already-prepared arrays (or mdtraj
trajectories with hydrogens stripped and topologies aligned). The
input preparation (load, strip H, align CA by residue index, superpose
onto the crystal frame) is the caller's job; that orchestration lives
in `premval.scoring`. Keeping the panel pure means each metric is
unit-testable on small synthetic inputs.

Metric correspondences with `analyze_ensembles.py` + `print_analysis.py`:

| premval                         | AlphaFlow source                                  |
|---------------------------------|---------------------------------------------------|
| `rmsf_correlation`              | `mdtraj.rmsf` + `correlations()`                  |
| `rmwd`                          | `get_mean_covar` + `sqrtm` + `RMWD` aggregation   |
| `md_pca_w2`                     | `get_pca` + `get_emd['ref|af']`                   |
| `contact_jaccard`               | `weak_contacts_iou` / `transient_contacts_iou`    |

AlphaFlow's `RAND1` reference subsample (seed 137) is preserved so per-
target numbers reproduce the paper. The seed is exposed as a parameter
in case downstream callers want different draws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from premval.metrics.alphaflow_port import get_mean_covar, get_pca, get_wasserstein, sqrtm

if TYPE_CHECKING:
    import mdtraj as md

ALPHAFLOW_SEED = 137
CONTACT_THRESHOLD_NM = 0.8
TRANSIENT_PROB_FLOOR = 0.1
WEAK_PROB_CEILING = 0.9


def rmsf_correlation(
    ref_traj: md.Trajectory,
    sub_traj: md.Trajectory,
    crystal_traj: md.Trajectory,
) -> dict[str, float | NDArray[np.floating]]:
    """Per-target Pearson correlation between reference and submission RMSF.

    Mirrors AlphaFlow: per-atom RMSF (in A) computed against the crystal
    reference frame, then Pearson r between the two RMSF vectors. Inputs
    must share the same atom ordering and count; the caller is responsible
    for hydrogen stripping and topology alignment.

    Args:
        ref_traj: Reference ensemble (e.g. ATLAS MD), hydrogens stripped.
        sub_traj: Submission ensemble, hydrogens stripped.
        crystal_traj: Single-frame reference structure used as the RMSF
            origin (the deposited crystal, also hydrogens-stripped).

    Returns:
        Dict with keys `rmsf_pearson` (float), `ref_rmsf`, `sub_rmsf`
        (per-atom arrays in A).
    """
    import mdtraj as md

    ref_rmsf = md.rmsf(ref_traj, crystal_traj) * 10
    sub_rmsf = md.rmsf(sub_traj, crystal_traj) * 10
    r, _ = pearsonr(sub_rmsf, ref_rmsf)
    return {"rmsf_pearson": float(r), "ref_rmsf": ref_rmsf, "sub_rmsf": sub_rmsf}


def rmwd(
    ref_xyz: NDArray[np.floating],
    sub_xyz: NDArray[np.floating],
    *,
    reference_subsample_size: int | None = None,
    seed: int = ALPHAFLOW_SEED,
) -> dict[str, float]:
    """Root-mean Wasserstein distance between two ensembles.

    Per atom, computes the 2-Wasserstein distance between two 3-D Gaussian
    approximations: translation component = ||mean_ref - mean_sub|| (A),
    fluctuation component = sqrt(tr(C_ref + C_sub - 2 * sqrt(C_ref @ C_sub)))
    (A). The atom-level values are RMS-aggregated, then combined via
    `rmwd = sqrt(emd_mean_rms^2 + emd_var_rms^2)`.

    The fallback for the sqrtm step (use `sqrt(trace(C_ref))` when the
    matrix square root fails or returns NaNs) is preserved from AlphaFlow.

    Args:
        ref_xyz: Reference coordinates `(n_ref_frames, n_atoms, 3)` (nm).
        sub_xyz: Submission coordinates `(n_sub_frames, n_atoms, 3)` (nm).
            `n_atoms` must match `ref_xyz`.
        reference_subsample_size: If set, subsample the reference to this
            many frames with `seed` before computing the moments. AlphaFlow
            uses 1000 for this step. If None, the full reference is used.
        seed: RNG seed for the reference subsample.

    Returns:
        Dict with keys `emd_mean_rms`, `emd_var_rms`, `rmwd` (all in A).
    """
    ref_arr = _maybe_subsample(ref_xyz, reference_subsample_size, seed)
    ref_mean, ref_covar = get_mean_covar(ref_arr)
    sub_mean, sub_covar = get_mean_covar(sub_xyz)

    emd_mean = (np.square(ref_mean - sub_mean).sum(-1) ** 0.5) * 10

    inner = ref_covar @ sub_covar
    try:
        root = sqrtm(inner)
        # tr(C_ref + C_sub - 2 sqrt(C_ref C_sub)) is the squared Bures
        # distance per atom; discard tiny imaginary parts from the
        # eigendecomp of the (non-symmetric) product, and clip to >=0
        # so floating-point noise around the identity case doesn't go
        # negative and produce NaNs under the sqrt.
        trace = np.trace(ref_covar + sub_covar - 2 * root, axis1=1, axis2=2).real
        emd_var = np.sqrt(np.clip(trace, 0.0, None)) * 10
        if not np.isfinite(emd_var).all():
            raise ValueError("non-finite emd_var")
    except (np.linalg.LinAlgError, ValueError):
        # AlphaFlow's published fallback (`trace(ref_covar)**0.5 * 10`)
        # collapses to a 3-element vector under their (n_atoms, 3, 3)
        # shape; we use the per-atom Cauchy-Schwarz upper bound
        # sqrt(tr(C_ref + C_sub)) * 10 instead so the rest of the pipeline
        # stays well-defined.
        fallback_trace = np.trace(ref_covar + sub_covar, axis1=1, axis2=2)
        emd_var = np.asarray(fallback_trace**0.5 * 10).real

    emd_mean_rms = float((np.square(emd_mean).mean()) ** 0.5)
    emd_var_rms = float((np.square(emd_var).mean()) ** 0.5)
    return {
        "emd_mean_rms": emd_mean_rms,
        "emd_var_rms": emd_var_rms,
        "rmwd": float(np.sqrt(emd_mean_rms**2 + emd_var_rms**2)),
    }


def md_pca_w2(
    ref_ca_xyz: NDArray[np.floating],
    sub_ca_xyz: NDArray[np.floating],
    *,
    k: int = 2,
    seed: int = ALPHAFLOW_SEED,
) -> float:
    """MD-PCA 2-Wasserstein in the reference PCA basis (first `k` components).

    Fit PCA on the reference CA coordinates, project both ensembles onto
    the first `k` components, build a square cost matrix of pairwise
    Cartesian distances (normalized by sqrt(n_residues) and scaled to A,
    matching AlphaFlow's `get_emd`), and return the 2-Wasserstein distance.
    The reference is subsampled to `sub_ca_xyz.shape[0]` frames so the
    cost matrix is square (Hungarian requires equal sample counts).

    Args:
        ref_ca_xyz: Reference CA coords `(n_ref_frames, n_residues, 3)`.
        sub_ca_xyz: Submission CA coords `(n_sub_frames, n_residues, 3)`.
        k: Number of PCA components to retain. Defaults to 2 (AlphaFlow's
            `K=2`).
        seed: RNG seed for the reference subsample.

    Returns:
        2-Wasserstein distance, A.
    """
    n_residues = ref_ca_xyz.shape[1]
    pca, ref_coords = get_pca(ref_ca_xyz)
    sub_coords = pca.transform(sub_ca_xyz.reshape(sub_ca_xyz.shape[0], -1))

    ref_subset = _index_subsample(ref_coords, sub_ca_xyz.shape[0], seed)

    ref_k = ref_subset[:, :k]
    sub_k = sub_coords[:, :k]
    distmat = np.square(ref_k[:, None] - sub_k[None]).sum(-1) ** 0.5
    distmat = distmat / n_residues**0.5 * 10
    return get_wasserstein(distmat, p=2)


def contact_jaccard(
    ref_ca_xyz: NDArray[np.floating],
    sub_ca_xyz: NDArray[np.floating],
    crystal_ca_xyz: NDArray[np.floating],
    *,
    contact_threshold_nm: float = CONTACT_THRESHOLD_NM,
    transient_prob_floor: float = TRANSIENT_PROB_FLOOR,
    weak_prob_ceiling: float = WEAK_PROB_CEILING,
    reference_subsample_size: int | None = None,
    seed: int = ALPHAFLOW_SEED,
) -> dict[str, float]:
    """Weak and transient contact Jaccard between reference and submission.

    Contacts are CA-CA pairs at distance < `contact_threshold_nm` in a
    frame. Per-residue-pair contact probability is the fraction of frames
    in which that pair contacts. The crystal frame defines a fixed contact
    mask; relative to it:

    - `weak`: crystal pairs that are present but loosely held (probability
      below `weak_prob_ceiling`). Separates collapsed/over-rigid models.
    - `transient`: non-crystal pairs that nonetheless form occasionally
      (probability above `transient_prob_floor`).

    Jaccard = |ref_mask ∩ sub_mask| / |ref_mask ∪ sub_mask|, computed
    independently for the weak and transient masks.

    Args:
        ref_ca_xyz: Reference CA coords `(n_ref_frames, n_residues, 3)`.
        sub_ca_xyz: Submission CA coords `(n_sub_frames, n_residues, 3)`.
        crystal_ca_xyz: Crystal CA coords `(n_residues, 3)`.
        contact_threshold_nm: Distance below which a pair is "in contact".
        transient_prob_floor: Minimum non-crystal contact probability to
            count as a transient contact.
        weak_prob_ceiling: Maximum crystal-pair contact probability to
            count as a weak contact.
        reference_subsample_size: If set, subsample the reference frames
            before computing the contact probability matrix.
        seed: RNG seed for the reference subsample.

    Returns:
        Dict with `weak_contacts_jaccard` and `transient_contacts_jaccard`
        as floats in [0, 1]. Either may be NaN if the corresponding
        denominator (union size) is zero.
    """
    ref_arr = _maybe_subsample(ref_ca_xyz, reference_subsample_size, seed)
    ref_distmat = np.linalg.norm(ref_arr[:, None, :] - ref_arr[:, :, None], axis=-1)
    sub_distmat = np.linalg.norm(sub_ca_xyz[:, None, :] - sub_ca_xyz[:, :, None], axis=-1)
    crystal_distmat = np.linalg.norm(
        crystal_ca_xyz[None, :] - crystal_ca_xyz[:, None], axis=-1
    )

    ref_prob = (ref_distmat < contact_threshold_nm).mean(0)
    sub_prob = (sub_distmat < contact_threshold_nm).mean(0)
    crystal_mask = crystal_distmat < contact_threshold_nm

    ref_weak = crystal_mask & (ref_prob < weak_prob_ceiling)
    sub_weak = crystal_mask & (sub_prob < weak_prob_ceiling)
    ref_transient = (~crystal_mask) & (ref_prob > transient_prob_floor)
    sub_transient = (~crystal_mask) & (sub_prob > transient_prob_floor)

    return {
        "weak_contacts_jaccard": _jaccard(ref_weak, sub_weak),
        "transient_contacts_jaccard": _jaccard(ref_transient, sub_transient),
    }


def _jaccard(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    union = (a | b).sum()
    if union == 0:
        return float("nan")
    return float((a & b).sum() / union)


def _maybe_subsample(
    arr: NDArray[np.floating], size: int | None, seed: int
) -> NDArray[np.floating]:
    if size is None or size >= arr.shape[0]:
        return arr
    return arr[_subsample_indices(arr.shape[0], size, seed)]


def _index_subsample(arr: NDArray[np.floating], size: int, seed: int) -> NDArray[np.floating]:
    if size >= arr.shape[0]:
        return arr
    return arr[_subsample_indices(arr.shape[0], size, seed)]


def _subsample_indices(n: int, size: int, seed: int) -> NDArray[np.intp]:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))
