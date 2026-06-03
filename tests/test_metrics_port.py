"""Tests for the AlphaFlow metric port.

The point is fidelity: the ported primitives must reproduce values that
match what AlphaFlow's original code would produce on the same inputs.
Tests use small synthetic arrays where the expected output is either
computed directly or has a closed-form check.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from premval.metrics.alphaflow_port import (
    get_mean_covar,
    get_pca,
    get_rmsds,
    get_wasserstein,
    sqrtm,
)


def _rng_xyz(n_frames: int, n_atoms: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float64)


def test_get_pca_shapes_and_round_trip() -> None:
    xyz = _rng_xyz(n_frames=20, n_atoms=5)
    pca, coords = get_pca(xyz)
    n_components = min(20, 5 * 3)
    assert coords.shape == (20, n_components)
    # Round-trip through PCA components should match original (centered) flat data.
    flat = xyz.reshape(20, -1)
    reconstructed = coords @ pca.components_ + pca.mean_
    np.testing.assert_allclose(reconstructed, flat, atol=1e-8)


def test_get_rmsds_per_frame_matches_manual() -> None:
    xyz1 = _rng_xyz(4, 7, seed=1)
    xyz2 = _rng_xyz(4, 7, seed=2)
    rmsds = get_rmsds(xyz1, xyz2)
    expected = np.linalg.norm(xyz1.reshape(4, -1) - xyz2.reshape(4, -1), axis=1)
    expected = expected / np.sqrt(7) * 10
    np.testing.assert_allclose(rmsds, expected)


def test_get_rmsds_broadcast_matches_cdist() -> None:
    xyz1 = _rng_xyz(3, 6, seed=3)
    xyz2 = _rng_xyz(5, 6, seed=4)
    rmsds = get_rmsds(xyz1, xyz2, broadcast=True)
    expected = cdist(xyz1.reshape(3, -1), xyz2.reshape(5, -1)) / np.sqrt(6) * 10
    np.testing.assert_allclose(rmsds, expected, atol=1e-10)


def test_get_mean_covar_shapes_and_values() -> None:
    xyz = _rng_xyz(50, 4, seed=5)
    mean, covar = get_mean_covar(xyz)
    assert mean.shape == (4, 3)
    assert covar.shape == (4, 3, 3)
    # For each atom, covar matches the population covariance of its frames.
    for atom_idx in range(4):
        centered = xyz[:, atom_idx, :] - mean[atom_idx]
        expected = (centered[:, :, None] * centered[:, None, :]).mean(0)
        np.testing.assert_allclose(covar[atom_idx], expected)


def test_sqrtm_squares_back_to_input() -> None:
    rng = np.random.default_rng(11)
    # Use a symmetric PSD matrix so the eigendecomp is real.
    base = rng.normal(size=(4, 4))
    psd = base @ base.T
    root = sqrtm(psd)
    np.testing.assert_allclose(np.asarray(root @ root).real, psd, atol=1e-8)


def test_get_wasserstein_zero_for_identical_distributions() -> None:
    # Empirical W_p between a sample and itself is 0 (optimal assignment
    # is the diagonal with zero cost).
    rng = np.random.default_rng(13)
    samples = rng.normal(size=(10, 2))
    distmat = cdist(samples, samples)
    assert get_wasserstein(distmat) == pytest.approx(0.0, abs=1e-12)


def test_get_wasserstein_p1_known_pairing() -> None:
    # Construct a deterministic matching: samples shifted by a constant.
    distmat = np.array(
        [
            [1.0, 5.0, 9.0],
            [5.0, 1.0, 5.0],
            [9.0, 5.0, 1.0],
        ]
    )
    w1 = get_wasserstein(distmat, p=1)
    assert w1 == pytest.approx(1.0)


def test_get_wasserstein_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square distmat"):
        get_wasserstein(np.zeros((2, 3)))
