"""`premval` command-line entry point.

Three subcommands:

- `premval fetch [--chains ...] [--kind ...]`: download ATLAS bundles
  into the local cache.
- `premval score --chain <id> --submission <pdb> [--out <json>]`: score
  one submission against one ATLAS reference and emit the metric panel
  as JSON.
- `premval prepare-refs [--chains ...] [--kind ...]`: precompute the
  per-target reference-observables cache (CA xyz, PCA, moments, contact
  prob, RMSF) for the val split or a chain list. Run after `fetch`.

The CLI is intentionally thin; orchestration logic lives in `scoring.py`
and `data.atlas` so it stays unit-testable without invoking argparse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from premval.data import (
    ATLAS_KINDS,
    AtlasKind,
    default_cache_dir,
    fetch_val_split,
    load_reference_observables,
    load_val_chains,
)
from premval.data.references import cache_path
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

    prep = sub.add_parser(
        "prepare-refs", help="precompute reference-observables cache for chains"
    )
    prep.add_argument(
        "--chains", nargs="*", default=None, help="chain ids; default = val split"
    )
    prep.add_argument(
        "--kind",
        choices=ATLAS_KINDS,
        default="analysis",
        help="ATLAS payload tier (default: analysis)",
    )
    prep.add_argument("--cache-dir", type=Path, default=None)
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
    for chain in chains:
        load_reference_observables(chain, kind=kind, cache_dir=cache_dir)
        print(f"{chain}\t{cache_path(chain, kind, cache_dir)}")
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
