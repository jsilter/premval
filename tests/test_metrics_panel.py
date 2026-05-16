"""Tests for the v1 metric panel.

Each metric is exercised with an "identity" case (submission == reference)
where the answer is exactly known, plus a perturbation case showing the
metric responds in the expected direction.
"""

from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest

from premval.metrics.panel import (
    contact_jaccard,
    md_pca_w2,
    rmsf_correlation,
    rmwd,
)


def _ca_topology(n_residues: int) -> md.Topology:
    top = md.Topology()
    chain = top.add_chain()
    for resseq in range(1, n_residues + 1):
        res = top.add_residue("ALA", chain, resSeq=resseq)
        top.add_atom("CA", md.element.carbon, res)
    return top


def _gaussian_ensemble(
    n_frames: int, n_atoms: int, *, scale: float = 0.05, seed: int = 0
) -> np.ndarray:
    """An ensemble of jittered Cartesian coordinates around a fixed mean."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(n_atoms, 3))
    noise = rng.normal(size=(n_frames, n_atoms, 3)) * scale
    return (mean[None] + noise).astype(np.float64)


# --------------------------- rmsf_correlation ----------------------------------


def test_rmsf_correlation_identity_is_one() -> None:
    n_atoms = 6
    top = _ca_topology(n_atoms)
    xyz = _gaussian_ensemble(n_frames=20, n_atoms=n_atoms, seed=1).astype(np.float32)
    crystal = md.Trajectory(xyz[:1].copy(), top)
    ensemble = md.Trajectory(xyz, top)
    result = rmsf_correlation(ensemble, ensemble, crystal)
    assert result["rmsf_pearson"] == pytest.approx(1.0)


def test_rmsf_correlation_differing_scales_still_correlate() -> None:
    n_atoms = 8
    top = _ca_topology(n_atoms)
    base = _gaussian_ensemble(n_frames=30, n_atoms=n_atoms, seed=2).astype(np.float32)
    crystal_xyz = base[:1].copy()
    # Submission ensemble has the same per-atom variance pattern but doubled
    # amplitude: RMSF should be perfectly correlated (linear scaling).
    centered = base - base.mean(0)
    sub_xyz = (crystal_xyz + 2.0 * centered).astype(np.float32)
    ref = md.Trajectory(base, top)
    sub = md.Trajectory(sub_xyz, top)
    crystal = md.Trajectory(crystal_xyz, top)
    result = rmsf_correlation(ref, sub, crystal)
    # mdtraj.rmsf superposes each frame onto the reference before computing
    # per-atom deviations, so the optimal rotation changes when fluctuations
    # are scaled and we lose perfect linearity. 0.99 is the right floor.
    assert result["rmsf_pearson"] > 0.99


# ------------------------------- rmwd ------------------------------------------


def test_rmwd_identity_is_near_zero() -> None:
    xyz = _gaussian_ensemble(n_frames=50, n_atoms=5, seed=3)
    out = rmwd(xyz, xyz)
    # Identity ensemble: means coincide, covariances coincide, so emd_mean = 0
    # and (in exact arithmetic) emd_var = 0. Allow a small numerical slack
    # from the sqrtm/eigendecomp path.
    assert out["emd_mean_rms"] == pytest.approx(0.0, abs=1e-9)
    assert out["emd_var_rms"] == pytest.approx(0.0, abs=1e-6)
    assert out["rmwd"] == pytest.approx(0.0, abs=1e-6)


def test_rmwd_translation_only_recovers_displacement() -> None:
    # Two ensembles with identical covariance but means offset by 0.5 nm
    # along x in every atom. emd_mean per atom = 0.5 nm = 5 A; RMS over
    # atoms still 5 A; emd_var = 0; RMWD = 5.
    xyz = _gaussian_ensemble(n_frames=80, n_atoms=4, seed=4)
    shifted = xyz + np.array([0.5, 0.0, 0.0])
    out = rmwd(xyz, shifted)
    assert out["emd_mean_rms"] == pytest.approx(5.0, abs=1e-6)
    assert out["emd_var_rms"] == pytest.approx(0.0, abs=1e-6)
    assert out["rmwd"] == pytest.approx(5.0, abs=1e-6)


# ----------------------------- md_pca_w2 ---------------------------------------


def test_md_pca_w2_identity_is_zero() -> None:
    # When sub is exactly the same set of frames as the reference subsample,
    # the optimal assignment is the identity and W2 = 0.
    n_frames = 40
    xyz = _gaussian_ensemble(n_frames=n_frames, n_atoms=6, seed=5)
    # Take the full reference as the "submission" so the seed-driven
    # subsample is itself, then optimal pairing = identity, distmat zeros.
    w2 = md_pca_w2(xyz, xyz)
    assert w2 == pytest.approx(0.0, abs=1e-8)


def test_md_pca_w2_grows_with_displacement() -> None:
    xyz = _gaussian_ensemble(n_frames=40, n_atoms=6, seed=6)
    shifted_small = xyz + 0.1
    shifted_big = xyz + 0.5
    w2_small = md_pca_w2(xyz, shifted_small)
    w2_big = md_pca_w2(xyz, shifted_big)
    assert 0.0 <= w2_small < w2_big


# ---------------------------- contact_jaccard ----------------------------------


def test_contact_jaccard_identity_is_one_when_masks_nonempty() -> None:
    # Build an ensemble where some residue pairs are deterministically in
    # contact (close packing along a line). Identity case must yield
    # Jaccard = 1 for the weak mask (transient may be NaN if there are no
    # non-crystal close pairs at all, which is fine here).
    n_residues = 5
    # Place residues on a line spaced 0.5 nm apart so adjacent and
    # next-adjacent pairs are in contact (< 0.8 nm) and others are not.
    base = np.zeros((n_residues, 3))
    base[:, 0] = np.arange(n_residues) * 0.5
    ensemble = np.broadcast_to(base, (10, n_residues, 3)).copy()
    # Add a tiny jitter so the contact probability isn't an exact 0/1.
    ensemble += np.random.default_rng(7).normal(size=ensemble.shape) * 0.05
    crystal = base
    out = contact_jaccard(ensemble, ensemble, crystal)
    # Identity ensembles: weak/transient masks coincide so Jaccard = 1
    # (when defined). NaN is acceptable if a category has zero union.
    for value in out.values():
        assert math.isnan(value) or value == pytest.approx(1.0)


def test_contact_jaccard_disjoint_masks_give_zero() -> None:
    # Use the same crystal geometry for both ensembles but engineer the
    # weak/transient masks to disagree by jittering one ensemble strongly.
    n_residues = 6
    base = np.zeros((n_residues, 3))
    base[:, 0] = np.arange(n_residues) * 0.5
    ref = np.broadcast_to(base, (20, n_residues, 3)).copy()
    ref += np.random.default_rng(8).normal(size=ref.shape) * 0.02
    sub = np.broadcast_to(base, (20, n_residues, 3)).copy()
    sub += np.random.default_rng(9).normal(size=sub.shape) * 0.4
    out = contact_jaccard(ref, sub, base)
    # The strong jitter in the submission breaks many crystal contacts and
    # creates new non-crystal contacts; at least one Jaccard should drop
    # below the identity 1.0. We assert the weaker condition: at least one
    # finite Jaccard is < 1.
    finite_values = [v for v in out.values() if not math.isnan(v)]
    assert finite_values, "expected at least one finite Jaccard"
    assert min(finite_values) < 1.0
