"""Generate ColabFold MSAs for ATLAS chains, in the layout AlphaFlow expects.

AlphaFlow (unlike ESMFlow / BioEmu / Str2Str / ConfDiff) is MSA-conditioned:
``predict.py --mode alphafold`` reads an alignment per chain at
``{msa_dir}/{chain}/a3m/{chain}.a3m`` (the path ``inference/alphaflow_run.py``
points ``ALPHAFLOW_MSA_DIR`` at). Those MSAs are *not* distributed anywhere:
the AlphaFlow HuggingFace repo ships only weights and sample ensembles, and the
upstream README says to generate them yourself. This script does that, querying
the public ColabFold MMseqs2 server -- the same path AlphaFlow's
``scripts/mmseqs_query.py`` uses -- so you can build MSAs on any machine (even
the CPU-only premval box) ahead of a GPU inference run.

Sequences come from premval's vendored split CSVs via
``premval.data.load_chain_sequences`` (the `seqres` column), so the chain ids
and residue sequences match exactly what the scorer and the AlphaFlow harness
use. premval stays CPU-only; this script needs only ``requests`` (+ optional
``tqdm``), no torch.

The MMseqs2 API client (``run_mmseqs2``) is vendored from ColabFold
(https://github.com/sokrypton/ColabFold, MIT) via AlphaFlow's
``scripts/mmseqs_query.py``, reduced to the monomer-MSA path this tool uses
(no templates, no chain pairing). The query parameters (``mode='env'``: paired
UniRef + environmental databases, redundancy-filtered) match AlphaFlow's call
so the generated alignments are equivalent to the published runs.

Output layout (resumable: existing ``.a3m`` files are skipped)::

    {out}/{chain}/a3m/{chain}.a3m

Run::

    python inference/atlas_msa.py --split val  --out /path/to/atlas_msa_dir
    python inference/atlas_msa.py --split test --out "$ALPHAFLOW_MSA_DIR" --chains 6o2v_A 5znj_A
    python inference/atlas_msa.py --self-test      # offline; no network

Then point the AlphaFlow harness at the result::

    export ALPHAFLOW_MSA_DIR=/path/to/atlas_msa_dir
    python inference/alphaflow_run.py --split val --checkpoint md
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import requests

logger = logging.getLogger("atlas_msa")

_COLABFOLD_HOST = "https://api.colabfold.com"
# ColabFold asks every caller to identify itself so the operators can reach out
# before rate-limiting a misbehaving client. Overridable via --user-agent.
_DEFAULT_USER_AGENT = "premval/0 (ATLAS MSA generation; https://github.com/jsilter/premval)"
# Submit sequences to the server in modest batches rather than one giant job:
# each completed batch is written to disk immediately, so an interrupted run
# resumes from where it stopped (existing .a3m files are skipped).
_DEFAULT_BATCH_SIZE = 25
_SUBMIT_TIMEOUT_S = 6.02  # ColabFold's recommended "multiple of 3, plus a bit"
_MAX_TRANSPORT_RETRIES = 5
# Pause between consecutive server jobs. The in-job poll loop already backs off
# on a RATELIMIT status, but submitting batch after batch with no gap is what
# trips the server's per-client limit in the first place; a fixed inter-batch
# delay keeps us comfortably under it. Only a handful of batches cover a split,
# so even a generous default costs little wall time.
_DEFAULT_BATCH_DELAY_S = 60.0


def _sleep(seconds: float) -> None:
    """Indirection so the offline self-test can monkeypatch out real sleeping."""
    time.sleep(seconds)


def run_mmseqs2(
    seqs: list[str],
    scratch_prefix: str,
    *,
    use_env: bool = True,
    use_filter: bool = True,
    host_url: str = _COLABFOLD_HOST,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> list[str]:
    """Query the ColabFold MMseqs2 server for monomer MSAs; return one a3m each.

    Vendored and trimmed from ColabFold's ``run_mmseqs2`` (MIT) via AlphaFlow's
    ``scripts/mmseqs_query.py``. Submits the sequences as one server job, polls
    until complete, downloads the result tarball, and concatenates the UniRef
    and (with ``use_env``) environmental a3m blocks per input sequence.

    Args:
        seqs: Query protein sequences (one-letter). Duplicates are de-duplicated
            for the server round-trip but a result is still returned per input.
        scratch_prefix: Path prefix for the server's scratch dir; the function
            creates ``{scratch_prefix}_{mode}`` and writes the tarball there.
        use_env: Include environmental databases (BFD/MGnify/...) alongside
            UniRef. Matches AlphaFlow's default.
        use_filter: Use the redundancy-filtered databases. Matches AlphaFlow's
            default.
        host_url: ColabFold API base URL.
        user_agent: Sent as the ``User-Agent`` header (ColabFold etiquette).

    Returns:
        List of a3m strings aligned to ``seqs`` (same length and order).

    Raises:
        RuntimeError: If the server reports ERROR/MAINTENANCE, or transport
            failures exceed the retry budget.
    """
    headers = {"User-Agent": user_agent}
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    def _request(method: str, path: str, **kwargs: object) -> requests.Response:
        errors = 0
        while True:
            try:
                return requests.request(
                    method,
                    f"{host_url}/{path}",
                    timeout=_SUBMIT_TIMEOUT_S,
                    headers=headers,
                    **kwargs,
                )
            except requests.exceptions.Timeout:
                logger.warning("timeout on %s %s; retrying", method, path)
            except requests.exceptions.RequestException as exc:
                errors += 1
                logger.warning("transport error (%d/%d): %s", errors, _MAX_TRANSPORT_RETRIES, exc)
                if errors >= _MAX_TRANSPORT_RETRIES:
                    raise RuntimeError(f"MMseqs2 API unreachable after {errors} retries") from exc
                _sleep(5)

    def _json(method: str, path: str, **kwargs: object) -> dict[str, str]:
        res = _request(method, path, **kwargs)
        try:
            out: dict[str, str] = res.json()
        except ValueError:
            logger.error("non-JSON reply from %s: %s", path, res.text[:200])
            return {"status": "ERROR"}
        return out

    # De-duplicate while preserving the first-seen order, and remember which
    # unique index each input maps to (so identical sequences share one query).
    unique: list[str] = []
    for seq in seqs:
        if seq not in unique:
            unique.append(seq)
    query = "".join(f">{101 + i}\n{seq}\n" for i, seq in enumerate(unique))

    scratch = Path(f"{scratch_prefix}_{mode}")
    scratch.mkdir(parents=True, exist_ok=True)
    tar_path = scratch / "out.tar.gz"

    if not tar_path.is_file():
        out = _json("POST", "ticket/msa", data={"q": query, "mode": mode})
        while out["status"] in ("UNKNOWN", "RATELIMIT"):
            _sleep(5 + random.randint(0, 5))
            out = _json("POST", "ticket/msa", data={"q": query, "mode": mode})
        if out["status"] in ("ERROR", "MAINTENANCE"):
            raise RuntimeError(f"MMseqs2 API status {out['status']}; try again later")
        ticket = out["id"]
        while out["status"] in ("UNKNOWN", "RUNNING", "PENDING"):
            _sleep(5 + random.randint(0, 5))
            out = _json("GET", f"ticket/{ticket}")
        if out["status"] != "COMPLETE":
            raise RuntimeError(f"MMseqs2 job ended in status {out['status']}")
        res = _request("GET", f"result/download/{ticket}")
        tar_path.write_bytes(res.content)

    a3m_files = [scratch / "uniref.a3m"]
    if use_env:
        a3m_files.append(scratch / "bfd.mgnify30.metaeuk30.smag30.a3m")
    if any(not f.is_file() for f in a3m_files):
        with tarfile.open(tar_path) as tar:
            # filter="data" rejects absolute paths / traversal / special files
            # (the ColabFold tarball is plain a3m/m8 regular files).
            tar.extractall(scratch, filter="data")

    # Gather a3m blocks keyed by the query index (the ">101", ">102", ... ids).
    # The server interleaves databases with NUL separators; a NUL resets the
    # "which query does this block belong to" tracking, mirroring ColabFold.
    blocks: dict[int, list[str]] = {}
    for a3m_file in a3m_files:
        update_key, key = True, None
        for line in a3m_file.read_text().splitlines(keepends=True):
            if "\x00" in line:
                line = line.replace("\x00", "")
                update_key = True
            if line.startswith(">") and update_key:
                key = int(line[1:].strip())
                update_key = False
                blocks.setdefault(key, [])
            if key is not None:
                blocks[key].append(line)

    per_unique = {i: "".join(blocks.get(101 + i, [])) for i in range(len(unique))}
    return [per_unique[unique.index(seq)] for seq in seqs]


def _msa_path(out_dir: Path, chain: str) -> Path:
    """Path AlphaFlow reads for a chain's alignment: `{out}/{chain}/a3m/{chain}.a3m`."""
    return out_dir / chain / "a3m" / f"{chain}.a3m"


def _pending_chains(sequences: dict[str, str], out_dir: Path, *, overwrite: bool) -> list[str]:
    """Chains still needing an MSA (file absent or empty), preserving CSV order."""
    pending = []
    for chain in sequences:
        path = _msa_path(out_dir, chain)
        if overwrite or not path.is_file() or path.stat().st_size == 0:
            pending.append(chain)
    return pending


def _progress(iterable: list[int], total: int, enabled: bool) -> object:
    """Wrap a range in tqdm when available and requested; else return it as-is."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="MSA batches", unit="batch")


def generate_msas(
    split: str,
    out_dir: Path,
    *,
    chains: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_delay: float = _DEFAULT_BATCH_DELAY_S,
    host_url: str = _COLABFOLD_HOST,
    user_agent: str = _DEFAULT_USER_AGENT,
    overwrite: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """Build ColabFold MSAs for an ATLAS split and write them under `out_dir`.

    Loads `{chain: seqres}` for `split`, restricts to `chains`/`limit` if given,
    skips chains that already have a non-empty a3m (unless `overwrite`), and
    queries the server in `batch_size`-sized jobs. Each batch is written before
    the next is submitted, so an interrupted run resumes cleanly.

    Args:
        split: `'val'` or `'test'`.
        out_dir: MSA root; chains land at `{out_dir}/{chain}/a3m/{chain}.a3m`.
        chains: Optional explicit chain subset (must exist in the split).
        limit: Optional cap on how many pending chains to process this run.
        batch_size: Sequences per ColabFold submission.
        batch_delay: Seconds to wait between consecutive server jobs, to stay
            under the public server's per-client rate limit. Not applied before
            the first batch or after the last.
        host_url: ColabFold API base URL.
        user_agent: `User-Agent` header value.
        overwrite: Rebuild even chains that already have an a3m.
        show_progress: Show a tqdm bar over batches when tqdm is installed.

    Returns:
        Paths of the a3m files written this run (excludes skipped ones).

    Raises:
        ValueError: If `split` is unknown or a requested chain is not in it.
    """
    from premval.data import load_chain_sequences

    sequences = load_chain_sequences(split)
    if chains is not None:
        missing = [c for c in chains if c not in sequences]
        if missing:
            raise ValueError(f"chains not in {split} split: {', '.join(missing)}")
        sequences = {c: sequences[c] for c in chains}

    pending = _pending_chains(sequences, out_dir, overwrite=overwrite)
    skipped = len(sequences) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    logger.info(
        "%s split: %d chains, %d already present, %d to build (batch=%d)",
        split,
        len(sequences),
        skipped,
        len(pending),
        batch_size,
    )
    if not pending:
        return []

    written: list[Path] = []
    batch_starts = list(range(0, len(pending), batch_size))
    for start in _progress(batch_starts, len(batch_starts), show_progress):
        batch = pending[start : start + batch_size]
        with tempfile.TemporaryDirectory(prefix="atlas_msa_") as scratch:
            a3ms = run_mmseqs2(
                [sequences[c] for c in batch],
                os.path.join(scratch, "q"),
                host_url=host_url,
                user_agent=user_agent,
            )
        for chain, a3m in zip(batch, a3ms, strict=True):
            path = _msa_path(out_dir, chain)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(a3m)
            written.append(path)
            logger.info("wrote %s (%d alignment lines)", path, a3m.count("\n"))
        if batch_delay > 0 and start != batch_starts[-1]:
            logger.info("sleeping %.0fs before next batch (rate-limit etiquette)", batch_delay)
            _sleep(batch_delay)
    return written


def _self_test() -> int:
    """Offline sanity check: sequence loading, path layout, a3m parsing. No network."""
    from premval.data import load_chain_sequences

    val = load_chain_sequences("val")
    test = load_chain_sequences("test")
    assert len(val) == 39, f"expected 39 val chains, got {len(val)}"
    assert len(test) == 82, f"expected 82 test chains, got {len(test)}"
    assert all(seq and seq.isalpha() for seq in val.values()), "empty/invalid seqres in val"

    try:
        load_chain_sequences("bogus")
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown split")

    sample = next(iter(test))
    assert _msa_path(Path("/tmp/msa"), sample) == Path(f"/tmp/msa/{sample}/a3m/{sample}.a3m")

    # `_pending_chains` should treat a present, non-empty file as done.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        first = next(iter(val))
        assert _pending_chains({first: val[first]}, out, overwrite=False) == [first]
        p = _msa_path(out, first)
        p.parent.mkdir(parents=True)
        p.write_text(">101\nABC\n")
        assert _pending_chains({first: val[first]}, out, overwrite=False) == []
        assert _pending_chains({first: val[first]}, out, overwrite=True) == [first]

    print("atlas_msa self-test OK")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ColabFold MSAs for ATLAS chains.")
    parser.add_argument("--split", choices=("val", "test"), help="ATLAS split to build MSAs for.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="MSA output root (default: $ALPHAFLOW_MSA_DIR).",
    )
    parser.add_argument("--chains", nargs="*", default=None, help="Optional explicit chain subset.")
    parser.add_argument("--limit", type=int, default=None, help="Cap pending chains this run.")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=_DEFAULT_BATCH_DELAY_S,
        help="seconds to pause between server jobs (rate-limit etiquette)",
    )
    parser.add_argument("--host-url", default=_COLABFOLD_HOST)
    parser.add_argument("--user-agent", default=_DEFAULT_USER_AGENT)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing MSAs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm bar.")
    parser.add_argument("--self-test", action="store_true", help="Offline check; no network.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.split is None:
        raise SystemExit("--split is required (or pass --self-test)")
    env_dir = os.environ.get("ALPHAFLOW_MSA_DIR")
    out_dir = args.out or (Path(env_dir) if env_dir else None)
    if out_dir is None:
        raise SystemExit("set --out or the ALPHAFLOW_MSA_DIR environment variable")

    written = generate_msas(
        args.split,
        out_dir,
        chains=args.chains,
        limit=args.limit,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
        host_url=args.host_url,
        user_agent=args.user_agent,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(f"wrote {len(written)} MSA(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
