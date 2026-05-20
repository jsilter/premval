"""Tests for premval.data.references module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from premval.data.references import (
    ReferenceObservables,
    _compute,
    _load_from_disk,
    _save,
    cache_path,
    load_reference_observables,
)
from tests.test_data import make_full_atom_trajectory


def _make_obs_from_traj(traj: object, tmp_path: Path) -> ReferenceObservables:
    """Compute reference observables from a trajectory using the real compute pipeline."""
    with patch("premval.data.references.load_chain_trajectory", return_value=traj):
        obs = _compute("test_A", "analysis", tmp_path)
    return obs


class TestRoundTrip:
    def test_save_load_values(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=40, n_residues=8)
        obs = _make_obs_from_traj(traj, tmp_path)

        path = tmp_path / "obs.npz"
        _save(obs, path)
        loaded = _load_from_disk(path)

        np.testing.assert_allclose(loaded.ref_xyz_ca, obs.ref_xyz_ca, atol=1e-6)
        np.testing.assert_allclose(loaded.crystal_xyz_ca, obs.crystal_xyz_ca, atol=1e-6)
        np.testing.assert_allclose(loaded.pca_components, obs.pca_components, atol=1e-6)
        np.testing.assert_allclose(loaded.pca_mean, obs.pca_mean, atol=1e-6)
        np.testing.assert_allclose(
            loaded.pca_explained_variance, obs.pca_explained_variance, atol=1e-6
        )
        np.testing.assert_allclose(loaded.ref_mean, obs.ref_mean, atol=1e-6)
        np.testing.assert_allclose(loaded.ref_covar, obs.ref_covar, atol=1e-6)
        np.testing.assert_allclose(loaded.ref_rmsf, obs.ref_rmsf, atol=1e-6)
        np.testing.assert_allclose(loaded.ref_contact_prob, obs.ref_contact_prob, atol=1e-6)
        np.testing.assert_array_equal(loaded.ca_indices, obs.ca_indices)

    def test_shapes(self, tmp_path: Path) -> None:
        n_residues = 10
        traj = make_full_atom_trajectory(n_frames=30, n_residues=n_residues)
        obs = _make_obs_from_traj(traj, tmp_path)

        assert obs.ca_indices.shape == (n_residues,)
        assert obs.ref_xyz_ca.shape == (30, n_residues, 3)
        assert obs.crystal_xyz_ca.shape == (n_residues, 3)
        assert obs.pca_mean.shape == (n_residues * 3,)
        assert obs.ref_mean.shape == (n_residues, 3)
        assert obs.ref_covar.shape == (n_residues, 3, 3)
        assert obs.ref_rmsf.shape == (n_residues,)
        assert obs.ref_contact_prob.shape == (n_residues, n_residues)

    def test_dtypes(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=20, n_residues=6)
        obs = _make_obs_from_traj(traj, tmp_path)

        assert obs.ref_xyz_ca.dtype == np.float32
        assert obs.crystal_xyz_ca.dtype == np.float32
        assert obs.pca_components.dtype == np.float32
        assert obs.pca_mean.dtype == np.float32
        assert obs.pca_explained_variance.dtype == np.float32
        assert obs.ref_mean.dtype == np.float32
        assert obs.ref_covar.dtype == np.float32
        assert obs.ref_rmsf.dtype == np.float32
        assert obs.ref_contact_prob.dtype == np.float32
        assert obs.ca_indices.dtype == np.int64

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        obs = _make_obs_from_traj(make_full_atom_trajectory(n_frames=20, n_residues=5), tmp_path)
        nested = tmp_path / "a" / "b" / "obs.npz"
        _save(obs, nested)
        assert nested.exists()


class TestCachePath:
    def test_layout(self, tmp_path: Path) -> None:
        path = cache_path("6cka_B", "analysis", tmp_path)
        assert path == tmp_path / "references" / "analysis" / "6cka_B.npz"

    def test_kind_separates_files(self, tmp_path: Path) -> None:
        a = cache_path("x", "analysis", tmp_path)
        p = cache_path("x", "protein", tmp_path)
        assert a != p
        assert a.parent != p.parent


class TestMissingCacheTriggersCompute:
    def test_first_call_creates_cache_file(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=25, n_residues=7)
        path = cache_path("mychain_A", "analysis", tmp_path)
        assert not path.exists()

        with patch("premval.data.references.load_chain_trajectory", return_value=traj):
            load_reference_observables("mychain_A", kind="analysis", cache_dir=tmp_path)

        assert path.exists()

    def test_second_call_loads_from_disk(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=25, n_residues=7)

        with patch(
            "premval.data.references.load_chain_trajectory", return_value=traj
        ) as mock_load:
            obs1 = load_reference_observables("mychain_A", kind="analysis", cache_dir=tmp_path)
            obs2 = load_reference_observables("mychain_A", kind="analysis", cache_dir=tmp_path)
            assert mock_load.call_count == 1

        np.testing.assert_allclose(obs1.ref_xyz_ca, obs2.ref_xyz_ca, atol=1e-6)

    def test_kind_scoped_separately(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=20, n_residues=6)
        with patch("premval.data.references.load_chain_trajectory", return_value=traj):
            load_reference_observables("chain_A", kind="analysis", cache_dir=tmp_path)
            load_reference_observables("chain_A", kind="protein", cache_dir=tmp_path)
        assert cache_path("chain_A", "analysis", tmp_path).exists()
        assert cache_path("chain_A", "protein", tmp_path).exists()

    def test_force_recomputes_and_overwrites_cache(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=25, n_residues=7)
        with patch(
            "premval.data.references.load_chain_trajectory", return_value=traj
        ) as mock_load:
            load_reference_observables("mychain_A", kind="analysis", cache_dir=tmp_path)
            assert mock_load.call_count == 1
            # Without force: cache hit, no recompute.
            load_reference_observables("mychain_A", kind="analysis", cache_dir=tmp_path)
            assert mock_load.call_count == 1
            # With force: recompute even though the cache exists.
            load_reference_observables(
                "mychain_A", kind="analysis", cache_dir=tmp_path, force=True
            )
            assert mock_load.call_count == 2


class TestPCAReprojection:
    def test_reprojection_consistent(self, tmp_path: Path) -> None:
        """PCA loaded from cache reprojects held-out frames consistently."""
        rng = np.random.default_rng(99)
        traj = make_full_atom_trajectory(n_frames=40, n_residues=8)

        with patch("premval.data.references.load_chain_trajectory", return_value=traj):
            obs = load_reference_observables("chain_B", kind="analysis", cache_dir=tmp_path)

        n_res = obs.pca_mean.shape[0] // 3
        held_out = rng.standard_normal((5, n_res, 3)).astype(np.float32)
        flat = held_out.reshape(5, n_res * 3)

        projected = (flat - obs.pca_mean) @ obs.pca_components.T

        obs2 = load_reference_observables("chain_B", kind="analysis", cache_dir=tmp_path)
        projected2 = (flat - obs2.pca_mean) @ obs2.pca_components.T

        np.testing.assert_allclose(projected, projected2, atol=1e-5)

    def test_pca_components_orthonormal(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=50, n_residues=8)
        with patch("premval.data.references.load_chain_trajectory", return_value=traj):
            obs = load_reference_observables("chain_C", kind="analysis", cache_dir=tmp_path)

        gram = obs.pca_components @ obs.pca_components.T
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-4)

    def test_crystal_is_first_frame(self, tmp_path: Path) -> None:
        traj = make_full_atom_trajectory(n_frames=30, n_residues=6)
        with patch("premval.data.references.load_chain_trajectory", return_value=traj):
            obs = load_reference_observables("chain_D", kind="analysis", cache_dir=tmp_path)
        np.testing.assert_allclose(obs.crystal_xyz_ca, obs.ref_xyz_ca[0], atol=1e-6)
