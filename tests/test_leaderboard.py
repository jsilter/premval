"""Tests for premval.leaderboard (batch scoring into results/ JSON).

Reuses the synthetic ATLAS bundle factory and a hand-written sample
ensemble so the whole batch path runs on CPU with no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from premval.data import load_chain_trajectory, sample_path
from premval.leaderboard import (
    LEADERBOARD_MODELS,
    average_metrics,
    load_leaderboard,
    score_model,
    score_split,
    split_chains,
    write_results,
)
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


def test_score_model_folds_telemetry_wall_seconds(tmp_path: Path) -> None:
    cache_dir = tmp_path / "atlas"
    samples_dir = tmp_path / "samples"
    _write_fake_bundle(cache_dir, VAL_CHAIN, frames_per_replica=(20, 20, 20))
    _write_sample_from_reference(samples_dir, "modelA", VAL_CHAIN, cache_dir)
    # Telemetry sidecar next to the sample, as the GPU harnesses write it.
    sidecar = sample_path("modelA", VAL_CHAIN, samples_dir).with_suffix(".telemetry.json")
    sidecar.write_text(json.dumps({"wall_seconds": 42.0}), encoding="utf-8")

    payload = score_model(
        "modelA",
        [VAL_CHAIN],
        split="val",
        cache_dir=cache_dir,
        samples_dir=samples_dir,
        enforce_size=30,
    )
    assert payload["per_target"][VAL_CHAIN]["wall_seconds"] == pytest.approx(42.0)


def test_average_metrics_medians_wall_seconds() -> None:
    # Quality metrics are meaned; wall_seconds is medianed (right-skewed cost).
    # The 1000.0 outlier would drag a mean to ~258 but leaves the median at 30.
    per_target = {
        "a": {"rmwd": 1.0, "wall_seconds": 10.0},
        "b": {"rmwd": 3.0, "wall_seconds": 30.0},
        "c": {"rmwd": 5.0, "wall_seconds": 50.0},
        "d": {"rmwd": 7.0, "wall_seconds": 1000.0},
        "e": {"rmwd": 9.0},  # no timing recorded -> excluded from the median
    }
    avg = average_metrics(per_target)
    assert avg is not None
    assert avg["rmwd"] == pytest.approx(5.0)  # mean(1,3,5,7,9)
    assert avg["wall_seconds"] == pytest.approx(40.0)  # median(10,30,50,1000)
    assert avg["n_chains"] == 5


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


def _write_results_payload(
    results_dir: Path, model: str, split: str, rmwd_by_chain: dict[str, float]
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    per_target = {c: {"chain": c, "rmwd": v} for c, v in rmwd_by_chain.items()}
    payload = {
        "model": model,
        "split": split,
        "n_scored": len(per_target),
        "per_target": per_target,
    }
    (results_dir / f"{model}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_leaderboard_missing_dir_shows_unscored_roster(tmp_path: Path) -> None:
    # With no results yet, the roster still renders (all-"n/a" rows) so models
    # being scored appear up front.
    rows = load_leaderboard(tmp_path / "nope")
    assert {row["model"] for row in rows} == set(LEADERBOARD_MODELS)
    assert all(row["test"] is None and row["val"] is None for row in rows)


def test_load_leaderboard_ranks_scored_before_roster(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    # Score two roster models: esmflow_md_distilled (test rmwd avg 2.0) ranks
    # ahead of bioemu (test rmwd avg 6.0, plus a val payload). The unscored
    # roster models fall to the bottom.
    _write_results_payload(results_dir, "esmflow_md_distilled", "test", {"a": 1.0, "b": 3.0})
    _write_results_payload(results_dir, "bioemu", "test", {"a": 5.0, "b": 7.0})
    _write_results_payload(results_dir / "val", "bioemu", "val", {"a": 4.0})

    rows = load_leaderboard(results_dir)

    assert [row["model"] for row in rows[:2]] == ["esmflow_md_distilled", "bioemu"]
    assert rows[0]["test"]["rmwd"] == pytest.approx(2.0)
    assert rows[1]["test"]["rmwd"] == pytest.approx(6.0)
    assert rows[1]["val"]["rmwd"] == pytest.approx(4.0)
    # Roster models with no results render after the scored ones.
    assert {row["model"] for row in rows} == set(LEADERBOARD_MODELS)
    assert all(row["test"] is None for row in rows[2:])
