import numpy as np
import pytest

from premval.metrics.panel import (
    compute_contact_prob,
    compute_per_atom_stats,
    contact_jaccard,
    rmwd,
    subsample,
)


def test_subsample_fewer_than_n() -> None:
    xyz = np.zeros((50, 5, 3), dtype=np.float32)
    result = subsample(xyz, n=100)
    assert result.shape == xyz.shape


def test_subsample_more_than_n() -> None:
    xyz = np.zeros((200, 5, 3), dtype=np.float32)
    result = subsample(xyz, n=100)
    assert result.shape == (100, 5, 3)


def test_subsample_deterministic() -> None:
    xyz = np.arange(200 * 5 * 3, dtype=np.float32).reshape(200, 5, 3)
    a = subsample(xyz, n=50, seed=1)
    b = subsample(xyz, n=50, seed=1)
    np.testing.assert_array_equal(a, b)


def test_contact_prob_shape() -> None:
    xyz = np.random.default_rng(0).standard_normal((20, 8, 3)).astype(np.float32)
    prob = compute_contact_prob(xyz)
    assert prob.shape == (8, 8)
    assert prob.dtype == np.float32
    np.testing.assert_array_equal(np.diag(prob), 0.0)


def test_contact_prob_range() -> None:
    xyz = np.random.default_rng(0).standard_normal((20, 6, 3)).astype(np.float32)
    prob = compute_contact_prob(xyz)
    assert np.all(prob >= 0.0)
    assert np.all(prob <= 1.0)


def test_per_atom_stats_shape() -> None:
    xyz = np.random.default_rng(0).standard_normal((30, 7, 3)).astype(np.float32)
    mean, covar = compute_per_atom_stats(xyz)
    assert mean.shape == (7, 3)
    assert covar.shape == (21, 21)


def test_rmwd_identical() -> None:
    mean = np.zeros((5, 3), dtype=np.float32)
    covar = np.eye(15, dtype=np.float32)
    assert rmwd(mean, covar, mean, covar) == pytest.approx(0.0, abs=1e-6)


def test_contact_jaccard_identical() -> None:
    prob = np.eye(6, dtype=np.float32)
    assert contact_jaccard(prob, prob) == pytest.approx(1.0)


def test_contact_jaccard_disjoint() -> None:
    a = np.zeros((6, 6), dtype=np.float32)
    a[0, 1] = a[1, 0] = 1.0
    b = np.zeros((6, 6), dtype=np.float32)
    b[2, 3] = b[3, 2] = 1.0
    assert contact_jaccard(a, b) == pytest.approx(0.0)
