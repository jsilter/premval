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
        and `per_target` (chain id -> metric dict from `score`).
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
            per_target[chain] = score_chain(
                sample, chain, enforce_size=enforce_size, cache_dir=cache_root
            )
        except ValueError as exc:
            _LOG.warning("%s/%s: skipping (%s)", model, chain, exc)
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
