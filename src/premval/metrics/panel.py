from typing import cast

import numpy as np

ALPHAFLOW_SEED = 137


def subsample(xyz: np.ndarray, n: int = 1000, seed: int = ALPHAFLOW_SEED) -> np.ndarray:
    """
    Randomly subsample n frames from xyz (n_frames, n_atoms, 3).
    If n >= n_frames, return all frames unchanged.
    """
    n_frames = xyz.shape[0]
    if n >= n_frames:
        return xyz
    rng = np.random.default_rng(seed)
    idx: np.ndarray = np.asarray(rng.choice(n_frames, size=n, replace=False))
    idx.sort()
    return cast(np.ndarray, xyz[idx])


def compute_contact_prob(xyz_ca: np.ndarray, threshold_nm: float = 0.8) -> np.ndarray:
    """
    Compute pairwise contact probability matrix.

    Args:
        xyz_ca: (n_frames, n_residues, 3) CA coordinates in nm
        threshold_nm: distance threshold in nm

    Returns:
        (n_residues, n_residues) float32 contact probability matrix, diagonal zero
    """
    # xyz_ca: (n_frames, n_res, 3)
    # diff[f, i, j, :] = xyz_ca[f, i, :] - xyz_ca[f, j, :]
    diff = xyz_ca[:, :, np.newaxis, :] - xyz_ca[:, np.newaxis, :, :]  # (n_frames, n_res, n_res, 3)
    dist = np.sqrt(np.sum(diff**2, axis=-1))  # (n_frames, n_res, n_res)
    contact = dist < threshold_nm  # (n_frames, n_res, n_res)
    prob: np.ndarray = np.asarray(contact.mean(axis=0), dtype=np.float32)  # (n_res, n_res)
    np.fill_diagonal(prob, 0.0)
    return prob


def compute_per_atom_stats(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-atom mean and covariance over frames.

    Args:
        xyz: (n_frames, n_atoms, 3) coordinates

    Returns:
        mean: (n_atoms, 3) float32
        covar: (n_atoms*3, n_atoms*3) float32 covariance of flattened coords
    """
    n_frames, n_atoms, _ = xyz.shape
    mean = xyz.mean(axis=0).astype(np.float32)  # (n_atoms, 3)
    flat = xyz.reshape(n_frames, n_atoms * 3).astype(np.float32)
    covar = np.cov(flat, rowvar=False).astype(np.float32)  # (n_atoms*3, n_atoms*3)
    return mean, covar


def rmwd(
    query_mean: np.ndarray,
    query_covar: np.ndarray,
    ref_mean: np.ndarray,
    ref_covar: np.ndarray,
) -> float:
    """
    Root Mean Wasserstein Distance between two Gaussian ensembles.
    Bures metric: W2^2 = ||mu1 - mu2||^2 + Bures(Sigma1, Sigma2)
    Simplified: use squared Frobenius norm of difference of means and covariances.
    """
    mean_term = float(np.sum((query_mean.ravel() - ref_mean.ravel()) ** 2))
    cov_term = float(np.sum((query_covar - ref_covar) ** 2))
    return float(np.sqrt(mean_term + cov_term))


def md_pca_w2(
    query_xyz_ca: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    pca_explained_variance: np.ndarray,
) -> float:
    """
    Wasserstein-2 distance in PCA space between query and reference (implicit via PCA).
    Project query onto reference PCA, compare distribution.
    """
    n_frames = query_xyz_ca.shape[0]
    flat = query_xyz_ca.reshape(n_frames, -1)
    projected = (flat - pca_mean) @ pca_components.T  # (n_frames, n_components)
    # Variance of query projections vs reference (explained_variance)
    query_var = projected.var(axis=0)
    w2_sq = float(np.sum((np.sqrt(query_var) - np.sqrt(pca_explained_variance)) ** 2))
    return float(np.sqrt(w2_sq))


def contact_jaccard(
    query_contact_prob: np.ndarray,
    ref_contact_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Jaccard similarity of binarized contact probability matrices.
    """
    query_bin = query_contact_prob > threshold
    ref_bin = ref_contact_prob > threshold
    intersection = float(np.sum(query_bin & ref_bin))
    union = float(np.sum(query_bin | ref_bin))
    if union == 0.0:
        return 1.0
    return intersection / union
