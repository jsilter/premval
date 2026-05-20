"""Tests for premval.leaderboard (batch scoring into results/ JSON).

Reuses the synthetic ATLAS bundle factory and a hand-written sample
ensemble so the whole batch path runs on CPU with no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from premval.data import load_chain_trajectory, sample_path
from premval.leaderboard import score_model, score_split, split_chains, write_results
from tests.test_data import _write_fake_bundle

# A real val chain so split_chains("val") includes it and the val-split
# wiring is exercised end-to-end.
VAL_CHAIN = "6e33_A"


def _write_sample_from_reference(
    samples_dir: Path, model: str, chain: str, cache_dir: Path
) -> None:
    """Write a sample ensemble that reuses the reference's own coordinates.

    Scoring a chain against an ensemble drawn from its own reference keeps
    the metric panel finite and the test fast; correctness of the metrics
    themselves is covered in test_scoring.py.
    """
    ref = load_chain_trajectory(chain, cache_dir=cache_dir)
    path = sample_path(model, chain, samples_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.save_pdb(str(path))


def test_split_chains_known_and_unknown() -> None:
    assert VAL_CHAIN in split_chains("val")
    assert len(split_chains("test")) == 82
    with pytest.raises(ValueError, match="unknown split"):
        split_chains("nope")


def test_score_model_writes_per_target(tmp_path: Path) -> None:
    cache_dir = tmp_path / "atlas"
    samples_dir = tmp_path / "samples"
    _write_fake_bundle(cache_dir, VAL_CHAIN, frames_per_replica=(20, 20, 20))
    _write_sample_from_reference(samples_dir, "modelA", VAL_CHAIN, cache_dir)

    payload = score_model(
        "modelA",
        [VAL_CHAIN],
        split="val",
        cache_dir=cache_dir,
        samples_dir=samples_dir,
        enforce_size=30,
    )
    assert payload["model"] == "modelA"
    assert payload["split"] == "val"
    assert payload["n_scored"] == 1
    assert VAL_CHAIN in payload["per_target"]
    assert "rmsf_pearson" in payload["per_target"][VAL_CHAIN]


def test_score_model_skips_missing_inputs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "atlas"
    samples_dir = tmp_path / "samples"
    # Bundle exists but no sample for this chain -> skipped silently.
    _write_fake_bundle(cache_dir, VAL_CHAIN, frames_per_replica=(20, 20, 20))

    payload = score_model(
        "modelA",
        [VAL_CHAIN, "7xyz_A"],
        split="val",
        cache_dir=cache_dir,
        samples_dir=samples_dir,
    )
    assert payload["n_scored"] == 0
    assert payload["per_target"] == {}


def test_score_model_skips_undersized_ensemble(tmp_path: Path) -> None:
    cache_dir = tmp_path / "atlas"
    samples_dir = tmp_path / "samples"
    _write_fake_bundle(cache_dir, VAL_CHAIN, frames_per_replica=(5, 5, 5))  # 15 frames
    _write_sample_from_reference(samples_dir, "modelA", VAL_CHAIN, cache_dir)

    # Demand 250 frames; the 15-frame sample fails the contract and is skipped,
    # not fatal.
    payload = score_model(
        "modelA",
        [VAL_CHAIN],
        split="val",
        cache_dir=cache_dir,
        samples_dir=samples_dir,
        enforce_size=250,
    )
    assert payload["n_scored"] == 0


def test_write_results_roundtrips(tmp_path: Path) -> None:
    payload = {"model": "m", "split": "val", "n_scored": 0, "per_target": {}}
    path = write_results(payload, tmp_path / "results")
    assert path == tmp_path / "results" / "m.json"
    assert json.loads(path.read_text())["model"] == "m"


def test_score_split_writes_one_file_per_model(tmp_path: Path) -> None:
    cache_dir = tmp_path / "atlas"
    samples_dir = tmp_path / "samples"
    out_dir = tmp_path / "results"
    _write_fake_bundle(cache_dir, VAL_CHAIN, frames_per_replica=(20, 20, 20))
    _write_sample_from_reference(samples_dir, "modelA", VAL_CHAIN, cache_dir)
    _write_sample_from_reference(samples_dir, "modelB", VAL_CHAIN, cache_dir)

    written = score_split(
        ["modelA", "modelB"],
        "val",
        cache_dir=cache_dir,
        samples_dir=samples_dir,
        out_dir=out_dir,
        enforce_size=30,
    )
    assert set(written) == {"modelA", "modelB"}
    for model, path in written.items():
        body = json.loads(path.read_text())
        assert body["model"] == model
        assert body["n_scored"] == 1


def test_score_split_defaults_to_cached_models(tmp_path: Path) -> None:
    cache_dir = tmp_path / "atlas"
    samples_dir = tmp_path / "samples"
    out_dir = tmp_path / "results"
    _write_fake_bundle(cache_dir, VAL_CHAIN, frames_per_replica=(20, 20, 20))
    _write_sample_from_reference(samples_dir, "only_model", VAL_CHAIN, cache_dir)

    written = score_split(
        None,
        "val",
        cache_dir=cache_dir,
        samples_dir=samples_dir,
        out_dir=out_dir,
        enforce_size=30,
    )
    assert set(written) == {"only_model"}
