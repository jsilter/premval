"""Run BioEmu on ATLAS chains and drop ensembles into the PREMVAL samples cache.

BioEmu (https://github.com/microsoft/bioemu, MIT) is a sequence-conditioned
generator of equilibrium protein structure ensembles. Given a single-chain
sequence it emits a backbone ensemble. This harness samples ``--n-samples``
(default 250) structures per ATLAS chain, loads the ``topology.pdb`` +
``samples.xtc`` pair BioEmu writes into its output directory via mdtraj, and
writes a single multi-model PDB to the path PREMVAL's scorer expects
(``sample_path(out_model, chain, dir)``).

Setup (run-yourself, GPU required)::

    pip install bioemu

BioEmu downloads its weights automatically from HuggingFace (the BioEmu
checkpoint plus the ColabFold/AlphaFold params it embeds the sequence with) on
first use, and sampling needs a single CUDA GPU.

This script lives OUTSIDE the importable ``premval`` package on purpose: the
package stays CPU-only and free of GPU dependencies. All heavy imports (torch,
bioemu, ...) are therefore deferred into the sampling function so that the
module imports, and ``--self-test`` runs, with only the CPU ``premval`` install
present.

The sampling entrypoint is ``bioemu.sample.main(sequence, num_samples,
output_dir, ...)``, which writes ``topology.pdb`` + ``samples.xtc`` into
``output_dir``. BioEmu's default ``filter_samples=True`` culls unphysical frames
(bad bond lengths / steric clashes), so the saved ensemble may hold slightly
fewer than ``num_samples`` frames; that matches how BioEmu is meant to be used.

Run::

    python inference/bioemu_run.py --split val
    python inference/bioemu_run.py --split test --chains 6o2v_A
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/bioemu_run.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path

from common import track_sample, wandb_run, write_telemetry

from premval.data import (
    default_samples_dir,
    load_test_chains,
    load_val_chains,
    sample_path,
)

DEFAULT_OUT_MODEL = "bioemu"
DEFAULT_N_SAMPLES = 250

# BioEmu's default filtering culls unphysical frames, so a request for exactly
# N rarely survives at N. Over-request by this fraction up front to absorb the
# typical cull, then top up (via BioEmu's native resume) if still short.
_INITIAL_OVERSAMPLE = 1.2
# Refuse to chase the target past this multiple of N; a chain that can't yield N
# physical frames within this budget is surfaced rather than burning compute.
_MAX_OVERSAMPLE = 3.0
# BioEmu's batch_size_100 (default 10) sets the GPU-memory budget shared across
# all lengths; see _batch_size_100_for. Calibrated on an A10G (22 GB): the
# default peaks at ~1.3 GB, so this targets ~12x that to fill the card and batch
# long chains in parallel instead of one structure at a time.
_BATCH_SIZE_100 = 120


def _seqres(split: str) -> dict[str, str]:
    """Return ``{chain_name: sequence}`` from the vendored ATLAS CSV for a split."""
    fname = "atlas_test.csv" if split == "test" else "atlas_val.csv"
    text = resources.files("premval.data").joinpath(fname).read_text(encoding="utf-8")
    return {row["name"]: row["seqres"] for row in csv.DictReader(text.splitlines())}


def _batch_size_100_for(seq_len: int) -> int:
    """Pick BioEmu's ``batch_size_100`` to fill the GPU instead of idling it.

    BioEmu derives its per-batch count as ``int(batch_size_100 * (100/L)**2)``;
    the ``(100/L)**2`` factor deliberately holds ``batch_size * L**2`` (a proxy
    for GPU memory) constant across lengths, so a single ``batch_size_100`` value
    sets one memory budget for every chain. BioEmu's default (10) is tuned for a
    worst-case card and uses only ~1.3 GB of an A10G's 22 GB, which forces
    ``batch_size`` down to 1 for long chains (L > ~316) and serialises sampling.

    Raising the constant to ``_BATCH_SIZE_100`` scales that budget up uniformly
    (~10x the default, calibrated to peak well under 22 GB), giving much larger
    batches, biggest where lengths are short, never flooring to 0 for any ATLAS
    chain. The ``max(..., ceil((L/100)**2))`` guard keeps the per-batch count >= 1
    for any hypothetical chain beyond ~1095 residues; for ATLAS the constant
    always dominates.
    """
    return max(_BATCH_SIZE_100, math.ceil((seq_len / 100) ** 2))


def _sample_chain(
    sequence: str, n_samples: int, dest: Path, max_oversample: float = _MAX_OVERSAMPLE
) -> None:
    """Sample a BioEmu ensemble for one sequence and write it as a multi-model PDB.

    Imports torch/bioemu/mdtraj lazily so the module stays importable on a
    CPU-only install. BioEmu writes ``topology.pdb`` + ``samples.xtc`` into a
    temp directory; with its default ``filter_samples=True`` the trajectory
    holds only the physically-valid frames, so we over-request and top up (via
    BioEmu's native resume) until at least ``n_samples`` survive, then
    deterministically subsample down to exactly ``n_samples`` (the leaderboard
    contract; see ``premval.io.enforce_ensemble_size``). The final multi-model
    PDB is written at ``dest``.

    Args:
        sequence: Single-chain protein sequence (``seqres``).
        n_samples: Number of structures the saved ensemble must contain.
        dest: Output path for the multi-model PDB. Parent must already exist.
        max_oversample: Refuse to chase the target past this multiple of
            ``n_samples``. Raise the default (``_MAX_OVERSAMPLE``) to rescue
            chains whose physical-frame survival is so low that the default cap
            can't reach ``n_samples`` (e.g. ~50x for a 3.6%-survival chain).

    Raises:
        RuntimeError: If BioEmu cannot yield ``n_samples`` physical frames
            within ``max_oversample`` times the request.
    """
    import mdtraj
    from bioemu.sample import main as bioemu_sample

    from premval.io import enforce_ensemble_size

    cap = int(n_samples * max_oversample)
    requested = int(n_samples * _INITIAL_OVERSAMPLE)
    batch_size_100 = _batch_size_100_for(len(sequence))
    with tempfile.TemporaryDirectory() as work:
        out_dir = Path(work)
        # Cache the (expensive) ColabFold/AlphaFold sequence embedding so it is
        # computed once per chain instead of recomputed for every batch. Without
        # this, long chains at batch_size=1 recompute a ~Lx-residue AF2 embedding
        # hundreds of times and effectively never finish.
        embeds_dir = out_dir / "embeds"
        embeds_dir.mkdir()
        while True:
            bioemu_sample(
                sequence=sequence,
                num_samples=requested,
                output_dir=str(out_dir),
                batch_size_100=batch_size_100,
                cache_embeds_dir=str(embeds_dir),
            )
            traj = mdtraj.load(str(out_dir / "samples.xtc"), top=str(out_dir / "topology.pdb"))
            if traj.n_frames >= n_samples:
                break
            if requested >= cap:
                raise RuntimeError(
                    f"BioEmu yielded only {traj.n_frames} physical frames from {requested} "
                    f"samples; need {n_samples} and will not exceed {cap}."
                )
            # Scale the next request by the observed survival rate, with a small
            # margin so we clear the target rather than landing exactly on it.
            survival = traj.n_frames / requested
            requested = min(cap, int((n_samples + 15) / max(survival, 0.1)))
    enforce_ensemble_size(traj, expected=n_samples).save_pdb(str(dest))


def _run(
    split: str,
    chains: list[str],
    out_model: str,
    n_samples: int,
    samples_dir: Path,
    max_oversample: float = _MAX_OVERSAMPLE,
) -> None:
    """Sample each chain that has no cached output yet (resumable)."""
    seqres = _seqres(split)
    config = {"model": out_model, "split": split, "n_samples": n_samples}
    with wandb_run(out_model, split, config) as logger:
        for chain in chains:
            dest = sample_path(out_model, chain, samples_dir)
            if dest.exists():
                print(f"skip {chain}: {dest} already exists")
                continue
            if chain not in seqres:
                raise KeyError(f"chain {chain!r} not in {split} split CSV")
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"sample {chain}: {n_samples} structures -> {dest}")
            try:
                with track_sample(chain, n_samples) as sink:
                    _sample_chain(seqres[chain], n_samples, dest, max_oversample)
            except Exception as exc:  # noqa: BLE001
                # Batch isolation: a single chain that OOMs or can't reach the
                # required physical-frame count shouldn't kill the other 120.
                # The skip-if-exists rule lets a later re-run retry just this one.
                print(f"FAILED {chain}: {type(exc).__name__}: {exc}; continuing")
                continue
            telemetry = sink[0]
            write_telemetry(dest, telemetry)
            logger.log(telemetry)
            print(telemetry.summary())


def _self_test(split: str, n_samples: int, samples_dir: Path) -> None:
    """Exercise the I/O path with synthetic frames, no GPU or BioEmu needed.

    For one or two chains, builds a CA-only ``mdtraj.Trajectory`` of exactly
    ``n_samples`` random frames sized to the chain's sequence length, writes it
    through the real ``sample_path`` location, reloads with
    ``premval.io.load_ensemble``, and asserts the frame count and path.
    """
    import mdtraj
    import numpy as np

    from premval.io import load_ensemble

    # The memory-budget constant dominates for every ATLAS length, and BioEmu's
    # int(batch_size_100*(100/L)**2) must never floor to 0 (its range() step).
    assert _batch_size_100_for(61) == _BATCH_SIZE_100
    assert _batch_size_100_for(724) == _BATCH_SIZE_100, "constant should dominate up to ~1095 res"
    for n_res in (61, 171, 300, 450, 724):
        assert int(_batch_size_100_for(n_res) * (100 / n_res) ** 2) >= 1, (
            f"batch size floors to 0 at L={n_res}"
        )

    seqres = _seqres(split)
    chains = list(seqres)[:2]
    for chain in chains:
        n_res = len(seqres[chain])
        topology = mdtraj.Topology()
        ca_chain = topology.add_chain()
        for _ in range(n_res):
            residue = topology.add_residue("ALA", ca_chain)
            topology.add_atom("CA", mdtraj.element.carbon, residue)
        xyz = np.random.default_rng(0).random((n_samples, n_res, 3)).astype("float32")
        traj = mdtraj.Trajectory(xyz, topology)

        dest = sample_path(DEFAULT_OUT_MODEL, chain, samples_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with track_sample(chain, n_samples) as sink:
            traj.save_pdb(str(dest))
        sidecar = write_telemetry(dest, sink[0])

        assert dest.exists(), f"{chain}: nothing written at expected path {dest}"
        reloaded = load_ensemble(dest)
        assert reloaded.n_frames == n_samples, f"{chain}: {reloaded.n_frames} != {n_samples}"
        assert sidecar.exists(), f"{chain}: no telemetry sidecar at {sidecar}"
        recorded = json.loads(sidecar.read_text())
        assert recorded["chain"] == chain and recorded["wall_seconds"] >= 0.0

    print(
        f"PASS self-test: {len(chains)} chain(s), {n_samples} frames each, wrote to {samples_dir}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--chains", nargs="+", help="chain ids; default is the whole split")
    parser.add_argument("--out-model", default=DEFAULT_OUT_MODEL)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--max-oversample",
        type=float,
        default=_MAX_OVERSAMPLE,
        help="cap on raw samples as a multiple of --n-samples; raise to rescue low-survival chains",
    )
    parser.add_argument("--samples-dir", type=Path, help="samples cache root")
    parser.add_argument("--self-test", action="store_true", help="run I/O self-test, no GPU")
    return parser.parse_args(argv)


def _resolve_samples_dir(arg: Path | None) -> Path:
    """Resolve the samples cache root: CLI flag, then ``PREMVAL_SAMPLES_DIR``, then default."""
    if arg is not None:
        return arg
    env = os.environ.get("PREMVAL_SAMPLES_DIR")
    return Path(env) if env else default_samples_dir()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples_dir = _resolve_samples_dir(args.samples_dir)

    if args.self_test:
        _self_test(args.split, args.n_samples, samples_dir)
        return 0

    chains = args.chains or (load_test_chains() if args.split == "test" else load_val_chains())
    _run(args.split, chains, args.out_model, args.n_samples, samples_dir, args.max_oversample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
