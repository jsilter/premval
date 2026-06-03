"""Faithful port of AlphaFlow's metric primitives.

Source: https://github.com/bjing2016/alphaflow, `scripts/analyze_ensembles.py`
(`get_pca`, `get_mean_covar`, `sqrtm`, `get_wasserstein`, `get_rmsds`). The
math is reproduced exactly so PREMVAL's panel reproduces the paper's
numbers; deviating from the recipe would void the port-fidelity check that
gates M1.

These are pure functions over numpy arrays. Higher-level metric assembly
(RMWD, MD-PCA W2, contact Jaccard) lives in `premval.scoring`.

Conventions inherited from AlphaFlow:
- xyz arrays are shape `(n_frames, n_atoms, 3)` in nanometers.
- All distance-style outputs are scaled by 10x to report Angstroms.
- RMSD per pair of frames is `||x - y||_2 / sqrt(n_atoms) * 10`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def get_pca(xyz: NDArray[np.floating]) -> tuple[PCA, NDArray[np.floating]]:
    """Fit a PCA on flattened per-frame coordinates and return projections.

    Replicates AlphaFlow's `get_pca`: reshapes `(n_frames, n_atoms, 3)` to
    `(n_frames, n_atoms * 3)` and fits a PCA with as many components as
    possible. The returned model can be used to project additional frames
    via `pca.transform(other.reshape(-1, n_atoms * 3))`.

    Args:
        xyz: Coordinates with shape `(n_frames, n_atoms, 3)` (nm).

    Returns:
        `(pca, coords)`. `coords` has shape `(n_frames, n_components)` and
        is the projection of `xyz` onto the fitted basis.
    """
    flat = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(flat.shape))
    coords = pca.fit_transform(flat)
    return pca, coords


def get_rmsds(
    xyz1: NDArray[np.floating],
    xyz2: NDArray[np.floating],
    *,
    broadcast: bool = False,
) -> NDArray[np.floating]:
    """Pairwise Cartesian-difference RMSD (no superposition), in Angstroms.

    AlphaFlow's `get_rmsds`. NOTE: this is *not* a rotation-aligned RMSD;
    it operates on whatever frame the inputs are already in, so callers
    typically superpose first.

    Args:
        xyz1: Shape `(n1, n_atoms, 3)`.
        xyz2: Shape `(n2, n_atoms, 3)`. `n_atoms` must match `xyz1`.
        broadcast: If True, returns the full `(n1, n2)` cross-distance
            matrix. If False, returns a per-frame `(n1,)` distance (requires
            `n1 == n2`).

    Returns:
        Distances in Angstroms; shape `(n1, n2)` if `broadcast` else `(n1,)`.
    """
    n_atoms = xyz1.shape[1]
    flat1 = xyz1.reshape(xyz1.shape[0], n_atoms * 3)
    flat2 = xyz2.reshape(xyz2.shape[0], n_atoms * 3)
    if broadcast:
        diff = flat1[:, None] - flat2[None]
    else:
        diff = flat1 - flat2
    rmsd = np.square(diff).sum(-1) ** 0.5 / n_atoms**0.5 * 10
    return np.asarray(rmsd)


def get_mean_covar(
    xyz: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Per-atom mean position and 3x3 covariance, the inputs to RMWD.

    AlphaFlow's `get_mean_covar`. Returns shapes `(n_atoms, 3)` for the
    mean and `(n_atoms, 3, 3)` for the covariance, computed across frames
    independently per atom.

    Args:
        xyz: Shape `(n_frames, n_atoms, 3)`.

    Returns:
        `(mean, covar)` with shapes `(n_atoms, 3)` and `(n_atoms, 3, 3)`.
    """
    mean = xyz.mean(0)
    centered = xyz - mean
    covar = (centered[..., None] * centered[..., None, :]).mean(0)
    return mean, covar


def sqrtm(matrix: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """Matrix square root via eigendecomposition (AlphaFlow's recipe).

    Used inside RMWD to square-root `ref_covar @ af_covar` per atom. For
    a symmetric PSD input the result is real and matches `scipy.linalg.sqrtm`
    closely; the product of two PSD matrices is not generally symmetric, so
    eigenvalues can be complex. Callers take the real part (or wrap in a
    try/except mirroring AlphaFlow's fallback to `trace(ref_covar)**0.5`).

    Args:
        matrix: Square matrix or batched `(..., n, n)` array.

    Returns:
        Matrix square root with the same shape; dtype may be complex.
    """
    eigvals, eigvecs = np.linalg.eig(matrix)
    root = (eigvecs * np.sqrt(eigvals[..., None, :])) @ np.linalg.inv(eigvecs)
    return np.asarray(root)


def get_wasserstein(distmat: NDArray[np.floating], *, p: int = 2) -> float:
    """`p`-Wasserstein distance between two equal-size empirical samples.

    AlphaFlow's `get_wasserstein`: given a square cost matrix of pairwise
    distances, solve the optimal-transport assignment via the Hungarian
    algorithm (`scipy.optimize.linear_sum_assignment`) and report the
    `p`-mean of the assigned distances. The matrix must be square because
    AlphaFlow's pipeline always feeds in equal sample counts; this is also
    why no `POT` dependency is needed.

    Args:
        distmat: Square `(n, n)` distance matrix (already in the units the
            caller wants reported).
        p: Order of the Wasserstein distance. Defaults to 2.

    Returns:
        The `p`-Wasserstein distance, a non-negative float.
    """
    if distmat.shape[0] != distmat.shape[1]:
        raise ValueError(
            f"get_wasserstein expects a square distmat; got shape {distmat.shape}."
        )
    cost = distmat**p
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean() ** (1 / p))
