"""Run BioEmu on ATLAS chains and drop ensembles into the PREMVAL samples cache.

BioEmu (https://github.com/microsoft/bioemu, MIT) is a sequence-conditioned
generator of equilibrium protein structure ensembles. Given a single-chain
sequence it emits a backbone ensemble; side-chain reconstruction yields an
all-atom topology (`.pdb`) plus a frame trajectory (`.xtc`). This harness
samples ``--n-samples`` (default 250) structures per ATLAS chain, converts the
``.pdb``/``.xtc`` pair into a single multi-model PDB via mdtraj, and writes it
to the path PREMVAL's scorer expects (``sample_path(out_model, chain, dir)``).

Setup (run-yourself, GPU required)::

    pip install bioemu

BioEmu downloads its weights automatically from HuggingFace
(``microsoft/bioemu``) on first use, and sampling needs a single CUDA GPU.
Side-chain reconstruction additionally pulls in BioEmu's optional
reconstruction extras the first time it runs.

This script lives OUTSIDE the importable ``premval`` package on purpose: the
package stays CPU-only and free of GPU dependencies. All heavy imports (torch,
bioemu, ...) are therefore deferred into the sampling function so that the
module imports, and ``--self-test`` runs, with only the CPU ``premval`` install
present.

Assumption: the BioEmu sampling entrypoint is ``bioemu.sample.main`` with the
keyword interface documented in the upstream README (``sequence``,
``num_samples``, ``output_dir``), and the all-atom run produces a ``samples.pdb``
topology plus a ``samples.xtc`` trajectory in ``output_dir``. The GPU run is
executed manually by the user, who can adjust these names if the installed
BioEmu version differs.

Run::

    python inference/bioemu_run.py --split val
    python inference/bioemu_run.py --split test --chains 6o2v_A
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/bioemu_run.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path

from premval.data import (
    default_samples_dir,
    load_test_chains,
    load_val_chains,
    sample_path,
)

DEFAULT_OUT_MODEL = "bioemu"
DEFAULT_N_SAMPLES = 250


def _seqres(split: str) -> dict[str, str]:
    """Return ``{chain_name: sequence}`` from the vendored ATLAS CSV for a split."""
    fname = "atlas_test.csv" if split == "test" else "atlas_val.csv"
    text = resources.files("premval.data").joinpath(fname).read_text(encoding="utf-8")
    return {row["name"]: row["seqres"] for row in csv.DictReader(text.splitlines())}


def _sample_chain(sequence: str, n_samples: int, dest: Path) -> None:
    """Sample a BioEmu ensemble for one sequence and write it as a multi-model PDB.

    Imports torch/bioemu/mdtraj lazily so the module stays importable on a
    CPU-only install. BioEmu writes a topology + trajectory pair into a temp
    directory; we load them with mdtraj and save a single multi-model PDB at
    ``dest``.

    Args:
        sequence: Single-chain protein sequence (``seqres``).
        n_samples: Number of structures to sample (one MODEL each).
        dest: Output path for the multi-model PDB. Parent must already exist.
    """
    import mdtraj
    from bioemu.sample import main as bioemu_sample

    with tempfile.TemporaryDirectory() as work:
        out_dir = Path(work)
        bioemu_sample(
            sequence=sequence,
            num_samples=n_samples,
            output_dir=str(out_dir),
        )
        traj = mdtraj.load(str(out_dir / "samples.xtc"), top=str(out_dir / "samples.pdb"))
    traj.save_pdb(str(dest))


def _run(split: str, chains: list[str], out_model: str, n_samples: int, samples_dir: Path) -> None:
    """Sample each chain that has no cached output yet (resumable)."""
    seqres = _seqres(split)
    for chain in chains:
        dest = sample_path(out_model, chain, samples_dir)
        if dest.exists():
            print(f"skip {chain}: {dest} already exists")
            continue
        if chain not in seqres:
            raise KeyError(f"chain {chain!r} not in {split} split CSV")
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"sample {chain}: {n_samples} structures -> {dest}")
        _sample_chain(seqres[chain], n_samples, dest)


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
        traj.save_pdb(str(dest))

        assert dest.exists(), f"{chain}: nothing written at expected path {dest}"
        reloaded = load_ensemble(dest)
        assert reloaded.n_frames == n_samples, f"{chain}: {reloaded.n_frames} != {n_samples}"

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
    _run(args.split, chains, args.out_model, args.n_samples, samples_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
