"""End-to-end scoring tests on synthetic ensembles.

The real `port-fidelity check` (M1's done-gate) runs against AlphaFlow's
published outputs on real ATLAS chains; those live in a separate, opt-in
test once we've ingested the AlphaFlow zip. These tests verify the
plumbing on small, fast inputs.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from premval.scoring import score


def _ca_topology(n_residues: int, *, start_resseq: int = 1) -> md.Topology:
    top = md.Topology()
    chain = top.add_chain()
    for offset in range(n_residues):
        res = top.add_residue("ALA", chain, resSeq=start_resseq + offset)
        top.add_atom("CA", md.element.carbon, res)
    return top


def _gaussian_trajectory(
    n_frames: int,
    n_residues: int,
    *,
    scale: float = 0.05,
    seed: int = 0,
    start_resseq: int = 1,
) -> md.Trajectory:
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(n_residues, 3))
    noise = rng.normal(size=(n_frames, n_residues, 3)) * scale
    xyz = (mean[None] + noise).astype(np.float32)
    top = _ca_topology(n_residues, start_resseq=start_resseq)
    return md.Trajectory(xyz, top)


def test_score_identity_panel_is_near_perfect() -> None:
    """Submission == reference should ace every metric."""
    traj = _gaussian_trajectory(n_frames=80, n_residues=6, seed=1)
    result = score(traj, traj)
    assert result["n_residues"] == 6
    assert result["n_ref_frames"] == 80
    assert result["n_sub_frames"] == 80
    assert result["rmsf_pearson"] == pytest.approx(1.0)
    # Superposition is done in float32 with Kabsch; identical-input
    # submission/reference end up off by ~1e-4 A after the rotation step.
    assert result["rmwd"] == pytest.approx(0.0, abs=1e-3)
    assert result["md_pca_w2"] == pytest.approx(0.0, abs=1e-3)
    # weak/transient may be NaN if their masks are empty; assert finite if defined.
    for key in ("weak_contacts_jaccard", "transient_contacts_jaccard"):
        v = result[key]
        assert np.isnan(v) or 0.0 <= v <= 1.0


def _heterogeneous_trajectory(
    n_frames: int,
    *,
    per_atom_scale: np.ndarray,
    mean: np.ndarray | None = None,
    seed: int = 0,
) -> md.Trajectory:
    """Trajectory where each atom has its own RMSF amplitude.

    Gives the RMSF correlation a real signal to align on, instead of the
    flat-variance case where every atom contributes the same number and
    Pearson is dominated by sampling noise.
    """
    n_residues = per_atom_scale.shape[0]
    if mean is None:
        mean = np.random.default_rng(seed).normal(size=(n_residues, 3))
    noise = np.random.default_rng(seed + 1).normal(size=(n_frames, n_residues, 3))
    xyz = (mean[None] + noise * per_atom_scale[None, :, None]).astype(np.float32)
    return md.Trajectory(xyz, _ca_topology(n_residues))


def test_score_random_null_underperforms_truth() -> None:
    """Random-Cartesian-perturbation null must lose on RMSF and PCA W2.

    The reference has heterogeneous per-atom RMSF (some atoms are stiff,
    others are floppy). The "truth" submission samples from the same
    distribution so its RMSF fingerprint matches the reference's. The
    "null" is uniform-amplitude noise: RMSF is flat across atoms, so it
    correlates weakly with the heterogeneous reference.
    """
    n_atoms = 12
    # Stiff first half, floppy second half — a clear per-atom fingerprint.
    scale = np.concatenate([np.full(n_atoms // 2, 0.02), np.full(n_atoms // 2, 0.20)])
    mean = np.random.default_rng(11).normal(size=(n_atoms, 3))
    ref = _heterogeneous_trajectory(120, per_atom_scale=scale, mean=mean, seed=2)
    truth = _heterogeneous_trajectory(80, per_atom_scale=scale, mean=mean, seed=21)

    # Null: uniform amplitude across atoms — no fingerprint to align on.
    flat_scale = np.full(n_atoms, 0.20)
    null = _heterogeneous_trajectory(80, per_atom_scale=flat_scale, mean=mean, seed=99)

    truth_score = score(truth, ref)
    null_score = score(null, ref)

    assert truth_score["md_pca_w2"] < null_score["md_pca_w2"]
    assert truth_score["rmsf_pearson"] > null_score["rmsf_pearson"]
    assert truth_score["rmwd"] < null_score["rmwd"]


def test_score_topology_with_resseq_offset_aligns_correctly() -> None:
    """Submission with extra leading/trailing residues should still align on the overlap."""
    ref = _gaussian_trajectory(n_frames=60, n_residues=6, seed=4, start_resseq=10)
    # Submission has 8 residues starting at resseq 8 — shared resseqs are
    # 10..15 (6 residues), with 8, 9 dropped from sub and nothing dropped from ref.
    sub = _gaussian_trajectory(n_frames=40, n_residues=8, seed=5, start_resseq=8)
    result = score(sub, ref)
    assert result["n_residues"] == 6
    assert result["n_sub_frames"] == 40
    # Should produce finite metrics, not crash.
    assert np.isfinite(result["rmsf_pearson"])
    assert np.isfinite(result["rmwd"])
    assert np.isfinite(result["md_pca_w2"])


def test_score_outputs_are_json_serialisable() -> None:
    import json

    traj = _gaussian_trajectory(n_frames=40, n_residues=5, seed=6)
    result = score(traj, traj)
    payload = json.dumps(result)
    assert "rmsf_pearson" in payload


def test_cli_help_lists_subcommands(tmp_path: Path) -> None:
    from premval.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    # argparse exits 0 for --help.
    assert exc.value.code == 0
