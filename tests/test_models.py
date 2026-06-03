"""Tests for the UI model-display registry."""

from __future__ import annotations

from premval.models import MODEL_REGISTRY, model_info


def test_model_info_known_flow_family() -> None:
    info = model_info("esmflow_md_distilled")
    assert info.name == "ESMFlow-MD (distilled)"
    assert info.url == "https://arxiv.org/abs/2402.04845"
    assert info.description is not None and "flow matching" in info.description
    assert info.citation is not None and "arXiv:2402.04845" in info.citation


def test_model_info_bioemu_proper_cased() -> None:
    info = model_info("bioemu")
    assert info.name == "BioEmu"
    assert info.url is not None
    assert info.description is not None
    assert info.citation is not None and "Lewis" in info.citation


def test_model_info_unknown_falls_back_to_key() -> None:
    info = model_info("mystery_model")
    assert info.name == "mystery_model"
    assert info.url is None
    assert info.description is None
    assert info.citation is None


def test_flow_family_covers_all_variants() -> None:
    for fam in ("alphaflow", "esmflow"):
        for tier in ("md", "pdb"):
            for variant in ("base", "distilled"):
                assert f"{fam}_{tier}_{variant}" in MODEL_REGISTRY


def test_md_variants_are_weak_holdout() -> None:
    for fam in ("alphaflow", "esmflow"):
        for variant in ("base", "distilled"):
            info = model_info(f"{fam}_md_{variant}")
            assert info.contamination == "weak_holdout"
            assert "ATLAS" in info.contamination_basis


def test_pdb_variants_and_bioemu_are_held_out() -> None:
    for fam in ("alphaflow", "esmflow"):
        for variant in ("base", "distilled"):
            assert model_info(f"{fam}_pdb_{variant}").contamination == "held_out"
    bioemu = model_info("bioemu")
    assert bioemu.contamination == "held_out"
    assert "40%" in bioemu.contamination_basis


def test_unknown_model_contamination_is_uncertain() -> None:
    info = model_info("mystery_model")
    assert info.contamination == "uncertain"
    assert info.contamination_basis
