"""`premval` command-line entry point.

Subcommands:

- `premval fetch [--chains ...] [--kind ...]`: download ATLAS bundles
  into the local cache.
- `premval score --chain <id> --submission <pdb> [--out <json>]`: score
  one submission against one ATLAS reference and emit the metric panel
  as JSON.
- `premval prepare-refs [--chains ...] [--kind ...]`: precompute the
  per-target reference-observables cache (CA xyz, PCA, moments, contact
  prob, RMSF) and the frame-trimmed reference view PDB the viewer streams,
  for the val split or a chain list. Run after `fetch`.
- `premval prepare-samples [--models ...] [--kind ...]`: precompute the
  per-(model, chain) viewer sidecars (aligned/view PDBs, sample
  observables, scored metrics) so the dashboard serves them without
  on-request computation. Run after `ingest` + `prepare-refs`.
- `premval serve`: run the FastAPI dashboard (requires the `[web]` extra).

The CLI is intentionally thin; orchestration logic lives in `scoring.py`
and `data.atlas` so it stays unit-testable without invoking argparse.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from premval.data import (
    ATLAS_KINDS,
    PUBLISHED_SOURCES,
    AtlasKind,
    default_cache_dir,
    fetch_published,
    fetch_val_split,
    load_reference_observables,
    load_val_chains,
    load_view_ensemble_pdb_bytes,
    view_ensemble_path,
    warm_view_caches,
)
from premval.data.references import cache_path
from premval.leaderboard import score_split
from premval.scoring import score_chain


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "prepare-refs":
        return _cmd_prepare_refs(args)
    if args.command == "prepare-samples":
        return _cmd_prepare_samples(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "score-all":
        return _cmd_score_all(args)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover - argparse guards


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="premval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download ATLAS bundles into the local cache")
    fetch.add_argument("--chains", nargs="*", default=None, help="chain ids; default = val split")
    fetch.add_argument(
        "--kind",
        choices=ATLAS_KINDS,
        default="analysis",
        help="ATLAS payload tier (default: analysis)",
    )
    fetch.add_argument("--cache-dir", type=Path, default=None)
    fetch.add_argument("--force", action="store_true", help="re-download cached bundles")

    score = sub.add_parser("score", help="score one submission against one ATLAS reference")
    score.add_argument("--chain", required=True, help="ATLAS chain id, e.g. 6cka_B")
    score.add_argument("--submission", required=True, type=Path, help="multi-model PDB path")
    score.add_argument(
        "--enforce-size",
        type=int,
        default=None,
        help="require this many frames in the submission (subsample if larger)",
    )
    score.add_argument("--cache-dir", type=Path, default=None)
    score.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write JSON to this path; otherwise stdout",
    )

    prep = sub.add_parser("prepare-refs", help="precompute reference-observables cache for chains")
    prep.add_argument("--chains", nargs="*", default=None, help="chain ids; default = val split")
    prep.add_argument(
        "--kind",
        choices=ATLAS_KINDS,
        default="analysis",
        help="ATLAS payload tier (default: analysis)",
    )
    prep.add_argument("--cache-dir", type=Path, default=None)
    prep.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute and overwrite chains that already have a cached .npz",
    )

    prep_samples = sub.add_parser(
        "prepare-samples", help="precompute per-(model, chain) viewer sidecars for the dashboard"
    )
    prep_samples.add_argument(
        "--models", nargs="*", default=None, help="model keys; default = all cached samples"
    )
    prep_samples.add_argument(
        "--kind",
        choices=ATLAS_KINDS,
        default="analysis",
        help="ATLAS payload tier of the reference bundles (default: analysis)",
    )
    prep_samples.add_argument("--cache-dir", type=Path, default=None)
    prep_samples.add_argument("--samples-dir", type=Path, default=None)
    prep_samples.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute and overwrite pairs that already have cached sidecars",
    )

    ingest = sub.add_parser(
        "ingest", help="download a published model's ATLAS ensembles into the samples cache"
    )
    ingest.add_argument(
        "--model",
        required=True,
        choices=sorted(PUBLISHED_SOURCES),
        help="published model key to ingest",
    )
    ingest.add_argument(
        "--chains", nargs="*", default=None, help="chain ids; default = all targets in the archive"
    )
    ingest.add_argument("--samples-dir", type=Path, default=None)
    ingest.add_argument(
        "--force",
        action="store_true",
        help="re-download the archive and overwrite cached ensembles",
    )

    score_all = sub.add_parser(
        "score-all", help="batch-score cached models across a split into results/"
    )
    score_all.add_argument(
        "--models", nargs="*", default=None, help="model keys; default = all cached samples"
    )
    score_all.add_argument(
        "--split", choices=("val", "test"), default="test", help="ATLAS split (default: test)"
    )
    score_all.add_argument("--cache-dir", type=Path, default=None)
    score_all.add_argument("--samples-dir", type=Path, default=None)
    score_all.add_argument(
        "--kind",
        choices=ATLAS_KINDS,
        default="analysis",
        help="ATLAS payload tier of the reference bundles (default: analysis)",
    )
    score_all.add_argument(
        "--out", type=Path, default=None, help="results directory (default: results/)"
    )

    serve = sub.add_parser("serve", help="run the FastAPI dashboard (requires [web] extra)")
    serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    serve.add_argument("--cache-dir", type=Path, default=None)
    serve.add_argument(
        "--kind",
        choices=ATLAS_KINDS,
        default="analysis",
        help="ATLAS payload tier (default: analysis)",
    )
    serve.add_argument("--reload", action="store_true", help="auto-reload on code change")
    return parser


def _cmd_fetch(args: argparse.Namespace) -> int:
    kind: AtlasKind = args.kind
    results = fetch_val_split(
        cache_dir=args.cache_dir,
        kind=kind,
        chains=args.chains,
        force=args.force,
    )
    for chain, path in results.items():
        print(f"{chain}\t{path}")
    return 0


def _cmd_prepare_refs(args: argparse.Namespace) -> int:
    kind: AtlasKind = args.kind
    chains = args.chains if args.chains else load_val_chains()
    cache_dir = args.cache_dir or default_cache_dir()
    bar = tqdm(chains, desc="prepare-refs", unit="chain")
    for chain in bar:
        npz = cache_path(chain, kind, cache_dir)
        view = view_ensemble_path(chain, kind, cache_dir)
        if npz.exists() and view.exists() and not args.overwrite:
            bar.set_postfix_str(f"{chain}: cached")
            continue
        bar.set_postfix_str(f"{chain}: computing", refresh=True)
        load_reference_observables(chain, kind=kind, cache_dir=cache_dir, force=args.overwrite)
        load_view_ensemble_pdb_bytes(chain, kind=kind, cache_dir=cache_dir, force=args.overwrite)
        bar.set_postfix_str(f"{chain}: done")
    return 0


def _cmd_prepare_samples(args: argparse.Namespace) -> int:
    summary = warm_view_caches(
        args.models,
        cache_dir=args.cache_dir,
        samples_dir=args.samples_dir,
        kind=args.kind,
        overwrite=args.overwrite,
    )
    failed_total = 0
    for model, (built, skipped, failed) in sorted(summary.items()):
        print(f"{model}\tbuilt={built} skipped={skipped} failed={failed}")
        failed_total += failed
    return 1 if failed_total else 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    results = fetch_published(
        args.model,
        samples_dir=args.samples_dir,
        chains=args.chains,
        force=args.force,
    )
    for chain, path in sorted(results.items()):
        print(f"{chain}\t{path}")
    print(f"ingested {len(results)} chains for {args.model!r}", file=sys.stderr)
    return 0


def _cmd_score_all(args: argparse.Namespace) -> int:
    written = score_split(
        args.models,
        args.split,
        cache_dir=args.cache_dir,
        samples_dir=args.samples_dir,
        kind=args.kind,
        out_dir=args.out,
    )
    for model, path in sorted(written.items()):
        print(f"{model}\t{path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        sys.stderr.write(
            "the `serve` subcommand requires the [web] extra; "
            "install with: pip install -e '.[web]'\n"
        )
        raise SystemExit(2) from exc

    # Env vars (not factory kwargs) so reload subprocesses inherit them.
    if args.cache_dir is not None:
        os.environ["PREMVAL_CACHE_DIR"] = str(args.cache_dir)
    os.environ["PREMVAL_KIND"] = args.kind

    uvicorn.run(
        "premval.web.app:create_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=args.reload,
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    result = score_chain(
        args.submission,
        args.chain,
        enforce_size=args.enforce_size,
        cache_dir=args.cache_dir,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is None:
        print(payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
