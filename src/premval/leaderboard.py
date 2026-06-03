"""Batch-score cached model ensembles into committed `results/` JSON.

This is the orchestration layer over `scoring.score_chain`: it walks a
split (val or test), scores every cached model against every chain that
has *both* a cached sample ensemble and a cached ATLAS reference, and
writes one `results/{model}.json` per model for the static leaderboard to
read.

Missing inputs (a model that hasn't generated a given chain, an ATLAS
reference that hasn't been fetched, or an ensemble that fails the
250-frame contract) are logged and skipped, never fatal: a partial run
still produces a usable results file, and re-running fills the gaps.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from premval.data.atlas import (
    AtlasKind,
    bundle_path,
    default_cache_dir,
    load_test_chains,
    load_val_chains,
)
from premval.data.samples import available_models, default_samples_dir, sample_path
from premval.io import DEFAULT_ENSEMBLE_SIZE
from premval.scoring import score_chain

if TYPE_CHECKING:
    from collections.abc import Iterable

_LOG = logging.getLogger(__name__)

Split = str  # "val" | "test"
DEFAULT_RESULTS_DIR = Path("results")

_SPLIT_LOADERS = {"val": load_val_chains, "test": load_test_chains}

# The numeric panel metrics that are meaningful to average across a split.
# Context fields (`n_residues`, frame counts, `chain`) are deliberately
# excluded.
AVERAGEABLE_METRICS = (
    "rmsf_pearson",
    "emd_mean_rms",
    "emd_var_rms",
    "rmwd",
    "md_pca_w2",
    "weak_contacts_jaccard",
    "transient_contacts_jaccard",
)

# Per-chain inference wall time (seconds), recorded by the GPU harnesses as
# telemetry sidecars and folded into `per_target` by `score_model`. Summarized
# on the leaderboard as the MEDIAN time-per-chain, not the mean: runtime is
# dominated by sequence length, so a split's per-chain wall times are strongly
# right-skewed (a few long chains run 100x the short ones) and the mean reports
# a "typical" cost far above what most chains take. Kept separate from the
# quality panel above because it is a cost figure (and device-dependent), not a
# metric.
TIMING_METRIC = "wall_seconds"


def average_metrics(per_target: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Per-split summary of each panel metric over a results payload's `per_target`.

    Quality metrics (`AVERAGEABLE_METRICS`) are summarized by their mean;
    `wall_seconds` (inference cost) by its median, since per-chain runtime is
    right-skewed and the mean overstates the typical chain. Non-finite per-chain
    values (e.g. a NaN contact Jaccard from an empty union) are skipped per
    metric, so a metric's summary is taken over only the chains where it is
    defined. Returns None for an empty `per_target`.

    Args:
        per_target: Mapping of chain id to its metric dict (the `per_target`
            block of a `score_model` payload).

    Returns:
        Dict of `{metric: value}` for every metric with at least one finite
        value (mean for quality metrics, median for `wall_seconds` when timing
        was recorded), plus `n_chains` (the number of scored chains summarized
        over), or None if `per_target` is empty.
    """
    import math
    import statistics

    if not per_target:
        return None
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    wall_seconds: list[float] = []
    for metrics in per_target.values():
        for key in AVERAGEABLE_METRICS:
            value = metrics.get(key)
            if isinstance(value, int | float) and math.isfinite(value):
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        timing = metrics.get(TIMING_METRIC)
        if isinstance(timing, int | float) and math.isfinite(timing):
            wall_seconds.append(float(timing))
    summary: dict[str, Any] = {key: sums[key] / counts[key] for key in sums}
    if wall_seconds:
        summary[TIMING_METRIC] = statistics.median(wall_seconds)
    summary["n_chains"] = len(per_target)
    return summary


def load_split_averages(
    model: str, results_dir: Path | None = None
) -> dict[str, dict[str, Any] | None]:
    """Read committed leaderboard results and average a model's metrics per split.

    Scans `results_dir` recursively for `{model}.json` files (the layout mixes
    `results/{model}.json` and `results/val/{model}.json`), trusts the `split`
    field inside each payload rather than its path, and for each split keeps
    the most complete payload (largest `n_scored`). Missing directory or model
    yields `{"test": None, "val": None}` so callers can render "n/a" instead of
    failing.

    Args:
        model: Samples-cache model key, e.g. `alphaflow_md_base`.
        results_dir: Results root. Defaults to `results/`.

    Returns:
        `{"test": <avg dict or None>, "val": <avg dict or None>}`.
    """
    root = results_dir or DEFAULT_RESULTS_DIR
    out: dict[str, dict[str, Any] | None] = {"test": None, "val": None}
    if not root.exists():
        return out
    best: dict[str, tuple[int, dict[str, dict[str, Any]]]] = {}
    for path in root.rglob(f"{model}.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        split = payload.get("split")
        if payload.get("model") != model or split not in out:
            continue
        n_scored = int(payload.get("n_scored", 0))
        if split not in best or n_scored > best[split][0]:
            best[split] = (n_scored, payload.get("per_target", {}))
    for split, (_n, per_target) in best.items():
        out[split] = average_metrics(per_target)
    return out


def _sample_wall_seconds(sample: Path) -> float | None:
    """Inference wall time (s) for a chain, read from its telemetry sidecar.

    The GPU harnesses write a `{chain}.telemetry.json` sidecar next to each
    sample PDB (see `inference/common.py`). Returns its `wall_seconds`, or None
    when the sidecar is absent or unreadable; timing is optional and the
    leaderboard renders "n/a" without it.
    """
    sidecar = sample.with_suffix(".telemetry.json")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get(TIMING_METRIC)
    return float(value) if isinstance(value, int | float) else None


def _rmwd_rank(row: dict[str, Any]) -> float:
    """Sort key for the leaderboard: best available RMWD (test preferred), lower first.

    Models with no finite RMWD on either split sort last (`math.inf`).
    """
    import math

    for split in ("test", "val"):
        avg = row.get(split)
        value = avg.get("rmwd") if avg else None
        if isinstance(value, int | float) and math.isfinite(value):
            return float(value)
    return math.inf


# The models the leaderboard always shows a row for, even before they have a
# committed results file (the row renders "n/a" until scored). Any extra model
# found in `results/` is appended to this roster.
LEADERBOARD_MODELS = (
    "alphaflow_md_base",
    "alphaflow_md_distilled",
    "esmflow_md_base",
    "esmflow_md_distilled",
    "bioemu",
)


def load_leaderboard(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Ranked per-model averages for the landing-page leaderboard.

    Always includes the `LEADERBOARD_MODELS` roster (rows render "n/a" until
    scored), unioned with any other model that has a committed `{model}.json`
    under `results_dir`. Each model's splits are folded via
    `load_split_averages`. Rows are sorted by best available RMWD (test
    preferred, then val; lower is better); models with no finite RMWD (not yet
    scored) sort last, alphabetically.

    Args:
        results_dir: Results root. Defaults to `results/`.

    Returns:
        List of `{"model": str, "test": <avg dict or None>, "val": <avg dict
        or None>}`, sorted best-RMWD-first.
    """
    root = results_dir or DEFAULT_RESULTS_DIR
    models: set[str] = set(LEADERBOARD_MODELS)
    if root.exists():
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            name = payload.get("model")
            if isinstance(name, str):
                models.add(name)
    rows = [{"model": model, **load_split_averages(model, root)} for model in sorted(models)]
    rows.sort(key=lambda row: (_rmwd_rank(row), row["model"]))
    return rows


def split_chains(split: Split) -> list[str]:
    """Return the chain ids for `split` ("val" or "test")."""
    try:
        return _SPLIT_LOADERS[split]()
    except KeyError:
        valid = sorted(_SPLIT_LOADERS)
        raise ValueError(f"unknown split {split!r}; expected one of {valid}") from None


def score_model(
    model: str,
    chains: Iterable[str],
    *,
    split: Split,
    cache_dir: Path | None = None,
    samples_dir: Path | None = None,
    kind: AtlasKind = "analysis",
    enforce_size: int = DEFAULT_ENSEMBLE_SIZE,
) -> dict[str, Any]:
    """Score one model across `chains`, returning the results payload.

    A chain is scored only if it has both a cached sample
    (`{samples_dir}/{model}/{chain}.pdb`) and a cached ATLAS reference
    bundle. Chains missing either, or whose ensemble fails the frame
    contract / topology alignment, are logged and omitted from
    `per_target`.

    Args:
        model: Samples-cache model key.
        chains: Chain ids to attempt.
        split: Split label recorded in the payload (for provenance).
        cache_dir: ATLAS cache root. Defaults to `default_cache_dir()`.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        kind: ATLAS payload tier of the reference bundles.
        enforce_size: Required ensemble frame count (subsample if larger,
            skip the chain if smaller).

    Returns:
        Dict with `model`, `split`, `scored_at` (UTC ISO-8601), `n_scored`,
        and `per_target` (chain id -> metric dict from `score`, plus
        `wall_seconds` when a telemetry sidecar accompanies the sample).
    """
    cache_root = cache_dir or default_cache_dir()
    samples_root = samples_dir or default_samples_dir()
    per_target: dict[str, dict[str, Any]] = {}
    for chain in chains:
        sample = sample_path(model, chain, samples_root)
        if not sample.exists():
            continue
        if not bundle_path(cache_root, kind, chain).exists():
            _LOG.warning("%s/%s: ATLAS reference not cached; skipping", model, chain)
            continue
        try:
            result = score_chain(sample, chain, enforce_size=enforce_size, cache_dir=cache_root)
        except ValueError as exc:
            _LOG.warning("%s/%s: skipping (%s)", model, chain, exc)
            continue
        # Drop the absolute, machine-local sample path: it's noise (and leaks a
        # home dir) in the committed results, and the chain id already keys the
        # entry. Kept on `score_chain`'s own return for ad-hoc/CLI use.
        result.pop("submission_path", None)
        wall_seconds = _sample_wall_seconds(sample)
        if wall_seconds is not None:
            result[TIMING_METRIC] = wall_seconds
        per_target[chain] = result
    return {
        "model": model,
        "split": split,
        "scored_at": datetime.now(UTC).isoformat(),
        "n_scored": len(per_target),
        "per_target": per_target,
    }


def write_results(payload: dict[str, Any], out_dir: Path | None = None) -> Path:
    """Write a `score_model` payload to `{out_dir}/{model}.json`, returning the path."""
    out = out_dir or DEFAULT_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{payload['model']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def score_split(
    models: Iterable[str] | None = None,
    split: Split = "test",
    *,
    cache_dir: Path | None = None,
    samples_dir: Path | None = None,
    kind: AtlasKind = "analysis",
    out_dir: Path | None = None,
    enforce_size: int = DEFAULT_ENSEMBLE_SIZE,
) -> dict[str, Path]:
    """Score every model across a split and write one results file per model.

    Args:
        models: Model keys to score. Defaults to every model present in the
            samples cache (`available_models`).
        split: "val" or "test".
        cache_dir: ATLAS cache root. Defaults to `default_cache_dir()`.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        kind: ATLAS payload tier of the reference bundles.
        out_dir: Results directory. Defaults to `results/`.
        enforce_size: Required ensemble frame count.

    Returns:
        Mapping of model key to the written results JSON path.
    """
    samples_root = samples_dir or default_samples_dir()
    keys = list(models) if models is not None else available_models(samples_root)
    chains = split_chains(split)
    written: dict[str, Path] = {}
    for model in keys:
        payload = score_model(
            model,
            chains,
            split=split,
            cache_dir=cache_dir,
            samples_dir=samples_root,
            kind=kind,
            enforce_size=enforce_size,
        )
        written[model] = write_results(payload, out_dir)
        _LOG.info("%s: scored %d/%d chains", model, payload["n_scored"], len(chains))
    return written
