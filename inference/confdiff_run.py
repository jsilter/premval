"""Run ConfDiff to generate ATLAS conformational ensembles for PREMVAL scoring.

ConfDiff (https://github.com/bytedance/ConfDiff, Apache-2.0) is an SE(3)
diffusion model over protein backbone frames. It is sequence-conditioned and
built on ESMFold representations, sampling diverse backbone conformations for a
single input sequence. This harness drives ConfDiff once per ATLAS chain to
produce a 250-MODEL multi-model PDB and drops it into PREMVAL's samples cache,
where the existing scorer picks it up.

The importable ``premval`` package is CPU-only and must never depend on torch /
GPU code. This script therefore lives OUTSIDE the package (top-level
``inference/``) and ``premval`` never imports it. All heavy imports (torch, the
ConfDiff repo, ESMFold) are performed lazily INSIDE the sampling function, so
this module imports and ``--self-test`` runs with only the CPU ``premval``
install present.

Checkpoint variants and contamination implications
---------------------------------------------------
ConfDiff publishes weights on HuggingFace (``leowang17/ConfDiff``). Two are
exposed here via ``--checkpoint``:

- ``base``: the PDB-trained base model. It never saw ATLAS MD trajectories, so
  relative to the ATLAS evaluation it is held-out. Default out-model key
  ``confdiff``; contamination label ``test``.
- ``md``: the ATLAS-fine-tuned ``-MD`` checkpoint. It was trained on ATLAS MD
  data, so scoring it on the ATLAS val/test splits is in-distribution. Default
  out-model key ``confdiff_md``; contamination label ``in_distribution``.

The contamination labels are recorded here for the operator's bookkeeping; this
script only writes the ensemble PDBs. ``--out-model`` defaults from the
checkpoint but can be overridden.

ConfDiff setup (performed manually on a GPU host)
-------------------------------------------------
1. Clone the repo and create its environment::

       git clone https://github.com/bytedance/ConfDiff
       cd ConfDiff
       conda env create -f environment.yml      # or follow the repo README
       conda activate confdiff
       pip install -e .

2. Download weights from HuggingFace ``leowang17/ConfDiff`` (both the base PDB
   checkpoint and the ATLAS-fine-tuned ``-MD`` checkpoint), e.g.::

       huggingface-cli download leowang17/ConfDiff --local-dir ./weights

3. Point this script at the clone and the weights via environment variables::

       export CONFDIFF_REPO=/path/to/ConfDiff
       export CONFDIFF_BASE_CKPT=/path/to/weights/confdiff_base.ckpt
       export CONFDIFF_MD_CKPT=/path/to/weights/confdiff_md.ckpt

Inference invocation (assumption)
---------------------------------
The exact ConfDiff sampling entrypoint and CLI flags vary by repo revision, so
this harness drives the repo's own sampling script via ``subprocess`` rather
than importing ConfDiff internals: it writes the chain's sequence to a FASTA
file, invokes ConfDiff's sampler to emit per-conformation PDBs into a temp dir,
then loads those frames with mdtraj and re-emits a single 250-MODEL PDB at
PREMVAL's ``sample_path``. The subprocess command is assembled in
``_confdiff_command`` and is the one thing most likely to need adjustment for a
given ConfDiff revision; the operator runs the GPU step manually and edits that
function if the invocation differs.

Run command
-----------
::

    python inference/confdiff_run.py --split test --checkpoint md
    python inference/confdiff_run.py --split val --checkpoint base --chains 6o2v_A 7ead_A
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/confdiff_run.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from common import track_sample, write_telemetry

from premval.data import (
    default_samples_dir,
    load_test_chains,
    load_val_chains,
    sample_path,
)

if TYPE_CHECKING:
    import mdtraj as md

N_SAMPLES = 250

# Default out-model (samples-cache key) per --checkpoint choice. The matching
# contamination labels (base -> test, md -> in_distribution) are documented in
# the module docstring; the operator records them when scoring.
DEFAULT_OUT_MODEL: dict[str, str] = {"base": "confdiff", "md": "confdiff_md"}


def _seqres(split: str) -> dict[str, str]:
    """Return ``{chain: seqres}`` for a split from the vendored ATLAS CSV."""
    fname = "atlas_test.csv" if split == "test" else "atlas_val.csv"
    text = resources.files("premval.data").joinpath(fname).read_text(encoding="utf-8")
    return {row["name"]: row["seqres"] for row in csv.DictReader(text.splitlines())}


def _default_chains(split: str) -> list[str]:
    """Return the full chain list for a split."""
    return load_test_chains() if split == "test" else load_val_chains()


def _confdiff_command(
    repo: Path, ckpt: Path, fasta: Path, out_dir: Path, n_samples: int
) -> list[str]:
    """Build the ConfDiff sampling subprocess command.

    This is the invocation most likely to need adjustment for a given ConfDiff
    revision (see module docstring). It targets the repo's sampling entrypoint,
    passing the input sequence FASTA, the checkpoint, the per-conformation
    output directory, and the number of samples.
    """
    return [
        sys.executable,
        str(repo / "scripts" / "sample.py"),
        "--ckpt",
        str(ckpt),
        "--fasta",
        str(fasta),
        "--output_dir",
        str(out_dir),
        "--num_samples",
        str(n_samples),
    ]


def _load_conformers(out_dir: Path, n_samples: int) -> md.Trajectory:
    """Stack ConfDiff's per-conformation PDBs into one trajectory.

    ConfDiff writes one PDB per sampled conformation. We load them all and join
    into a single trajectory; ``enforce_ensemble_size`` then trims/checks to
    exactly ``n_samples`` frames.
    """
    import mdtraj as md

    from premval.io import enforce_ensemble_size

    pdbs = sorted(out_dir.glob("*.pdb"))
    if not pdbs:
        raise RuntimeError(f"ConfDiff produced no PDB conformers in {out_dir}")
    traj = md.join([md.load(str(p)) for p in pdbs])
    return enforce_ensemble_size(traj, expected=n_samples)


def _sample_chain(chain: str, seq: str, ckpt: Path, repo: Path, n_samples: int) -> md.Trajectory:
    """Generate ``n_samples`` conformations for one chain via ConfDiff.

    Heavy / GPU work lives entirely here (lazy imports only). Writes the
    sequence to a FASTA, runs ConfDiff's sampler in a temp dir, and returns the
    stacked, size-enforced trajectory.
    """
    with tempfile.TemporaryDirectory(prefix=f"confdiff_{chain}_") as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / f"{chain}.fasta"
        fasta.write_text(f">{chain}\n{seq}\n", encoding="utf-8")
        out_dir = tmp_path / "conformers"
        out_dir.mkdir()
        cmd = _confdiff_command(repo, ckpt, fasta, out_dir, n_samples)
        subprocess.run(cmd, check=True, cwd=str(repo))
        return _load_conformers(out_dir, n_samples)


def _checkpoint_path(checkpoint: str) -> Path:
    """Resolve the checkpoint file path from the environment."""
    env = "CONFDIFF_MD_CKPT" if checkpoint == "md" else "CONFDIFF_BASE_CKPT"
    value = os.environ.get(env)
    if not value:
        raise SystemExit(f"set {env} to the ConfDiff {checkpoint} checkpoint path")
    return Path(value)


def _repo_path() -> Path:
    """Resolve the ConfDiff repo clone path from the environment."""
    value = os.environ.get("CONFDIFF_REPO")
    if not value:
        raise SystemExit("set CONFDIFF_REPO to the ConfDiff repo clone path")
    return Path(value)


def _resolve_samples_dir(arg: Path | None) -> Path:
    """Resolve the samples cache root: CLI flag, then ``PREMVAL_SAMPLES_DIR``, then default."""
    if arg is not None:
        return arg
    env = os.environ.get("PREMVAL_SAMPLES_DIR")
    return Path(env) if env else default_samples_dir()


def run(
    *,
    split: str,
    chains: list[str],
    checkpoint: str,
    out_model: str,
    n_samples: int,
    samples_dir: Path,
) -> None:
    """Sample each chain with ConfDiff and write a 250-MODEL PDB to the cache.

    Resumable: a chain whose output PDB already exists is skipped.
    """
    seqres = _seqres(split)
    repo = _repo_path()
    ckpt = _checkpoint_path(checkpoint)

    for chain in chains:
        dest = sample_path(out_model, chain, samples_dir)
        if dest.exists():
            print(f"skip {chain}: {dest} exists")
            continue
        seq = seqres.get(chain)
        if seq is None:
            raise SystemExit(f"no seqres for chain {chain!r} in split {split!r}")
        print(f"sample {chain} ({len(seq)} residues) -> {dest}")
        with track_sample(chain, n_samples) as sink:
            traj = _sample_chain(chain, seq, ckpt, repo, n_samples)
        dest.parent.mkdir(parents=True, exist_ok=True)
        traj.save_pdb(str(dest))
        telemetry = sink[0]
        write_telemetry(dest, telemetry)
        print(f"wrote {dest} ({traj.n_frames} frames)")
        print(telemetry.summary())


def _synthetic_traj(n_residues: int, n_frames: int) -> md.Trajectory:
    """Build a CA-only trajectory of ``n_frames`` random frames.

    Used only by ``--self-test`` to exercise the real output path without any
    GPU/model work.
    """
    import mdtraj as md
    import numpy as np

    top = md.Topology()
    rng = np.random.default_rng(0)
    chain = top.add_chain()
    for i in range(n_residues):
        residue = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, residue)
    xyz = rng.standard_normal((n_frames, n_residues, 3)).astype("float32")
    return md.Trajectory(xyz=xyz, topology=top)


def _self_test(split: str, n_samples: int, samples_dir: Path) -> int:
    """Run the offline self-test: no GPU/model work. Returns an exit code."""
    from premval.io import load_ensemble

    for ck, expected in (("base", "confdiff"), ("md", "confdiff_md")):
        assert DEFAULT_OUT_MODEL[ck] == expected, f"out-model default for {ck} should be {expected}"

    seqres = _seqres(split)
    chains = _default_chains(split)[:2]
    out_model = DEFAULT_OUT_MODEL["base"]

    for chain in chains:
        seq = seqres[chain]
        traj = _synthetic_traj(len(seq), n_samples)
        dest = sample_path(out_model, chain, samples_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with track_sample(chain, n_samples) as sink:
            traj.save_pdb(str(dest))
        sidecar = write_telemetry(dest, sink[0])

        assert dest == sample_path(out_model, chain, samples_dir)
        reloaded = load_ensemble(dest)
        assert reloaded.n_frames == n_samples, (
            f"{chain}: expected {n_samples} frames, got {reloaded.n_frames}"
        )
        assert sidecar.exists(), f"{chain}: no telemetry sidecar at {sidecar}"
        recorded = json.loads(sidecar.read_text())
        assert recorded["chain"] == chain and recorded["wall_seconds"] >= 0.0
        print(f"  {chain}: {reloaded.n_frames} frames at {dest}")

    print(f"SELF-TEST PASS: {len(chains)} chains, {n_samples} frames each, out-model defaults OK")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ConfDiff to generate ATLAS ensembles.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--chains", nargs="+", default=None, help="Chain ids; default = whole split."
    )
    parser.add_argument("--checkpoint", choices=("base", "md"), default="base")
    parser.add_argument(
        "--out-model", default=None, help="Samples-cache key; default from --checkpoint."
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument(
        "--self-test", action="store_true", help="Offline check; no GPU/model work."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples_dir = _resolve_samples_dir(args.samples_dir)
    if args.self_test:
        return _self_test(args.split, args.n_samples, samples_dir)

    out_model = args.out_model or DEFAULT_OUT_MODEL[args.checkpoint]
    chains = args.chains if args.chains is not None else _default_chains(args.split)
    run(
        split=args.split,
        chains=chains,
        checkpoint=args.checkpoint,
        out_model=out_model,
        n_samples=args.n_samples,
        samples_dir=samples_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
