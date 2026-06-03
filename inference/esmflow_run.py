"""Run ESMFlow (distilled) to generate ATLAS conformational ensembles for PREMVAL.

ESMFlow is the ESMFold-backboned variant of AlphaFlow (Jing et al. 2024,
https://github.com/bjing2016/alphaflow, MIT). It is a flow-matching generator
that turns ESMFold into a sequence-conditioned ensemble sampler. This harness
drives ESMFlow once per ATLAS chain to produce a 250-MODEL multi-model PDB
and drops it into PREMVAL's samples cache, where the existing scorer picks
it up.

The importable ``premval`` package is CPU-only and must never depend on
torch / GPU code. This script therefore lives OUTSIDE the package (top-level
``inference/``) and ``premval`` never imports it. All heavy imports (torch,
mdtraj, the AlphaFlow/ESMFlow repo) are performed lazily INSIDE the sampling
function, so this module imports and ``--self-test`` runs with only the CPU
``premval`` install present.

Why ESMFlow shares the AlphaFlow repo
-------------------------------------
ESMFlow and AlphaFlow ship in the same upstream repo (``bjing2016/alphaflow``);
the only differences at inference time are ``--mode esmfold`` and a different
weights file. ESMFlow does *not* need MSAs (the ESMFold backbone is
single-sequence). The harnesses are kept separate for clarity, telemetry, and
out-model bookkeeping; both read ``ALPHAFLOW_REPO`` as the repo clone path.

Checkpoint variants and contamination implications
---------------------------------------------------
This harness defaults to the **distilled** variants (much cheaper than the
base diffusion checkpoints, comparable quality per the paper). Two are
exposed via ``--checkpoint``:

- ``pdb``: PDB-trained distilled. Never saw ATLAS MD trajectories, so
  ATLAS evaluation is held-out. Default out-model ``esmflow_pdb_distilled``;
  contamination label ``test``.
- ``md``: ATLAS-fine-tuned distilled. Trained on ATLAS MD, so scoring it on
  the ATLAS val/test splits is in-distribution. Default out-model
  ``esmflow_md_distilled``; contamination label ``in_distribution``.

ESMFlow setup (performed manually on a GPU host)
------------------------------------------------
1. Clone the AlphaFlow repo and create its environment (same as the
   AlphaFlow harness)::

       git clone https://github.com/bjing2016/alphaflow
       cd alphaflow
       conda env create -f environment.yml
       conda activate alphaflow

2. Download the distilled ESMFlow weights from HuggingFace
   ``bjing-mit/alphaflow``, e.g.::

       huggingface-cli download bjing-mit/alphaflow \\
           esmflow_pdb_distilled_202402.pt esmflow_md_distilled_202402.pt \\
           --local-dir ./weights

3. Point this script at the clone and weights via environment variables
   (no MSA dir required)::

       export ALPHAFLOW_REPO=/path/to/alphaflow
       export ESMFLOW_PDB_DISTILLED_CKPT=/path/to/weights/esmflow_pdb_distilled_202402.pt
       export ESMFLOW_MD_DISTILLED_CKPT=/path/to/weights/esmflow_md_distilled_202402.pt

Inference invocation
--------------------
This harness shells out to the repo's ``predict.py`` rather than importing
internals; ``_esmflow_command`` assembles the documented flags
(``--mode esmfold --noisy_first --no_diffusion --samples 250``, per the
AlphaFlow README ESMFlow + distilled sections). ``predict.py`` writes one
multi-model PDB per input row (``{outpdb}/{name}.pdb``); this harness loads
it and re-emits at ``sample_path(out_model, chain, samples_dir)`` after
verifying the frame count.

Run command
-----------
::

    python inference/esmflow_run.py --split val  --checkpoint pdb
    python inference/esmflow_run.py --split test --checkpoint md
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/esmflow_run.py --self-test
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

from common import track_sample, wandb_run, write_telemetry

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
# contamination labels (pdb -> test, md -> in_distribution) live in
# data/contamination_labels.yaml.
DEFAULT_OUT_MODEL: dict[str, str] = {
    "pdb": "esmflow_pdb_distilled",
    "md": "esmflow_md_distilled",
}


def _seqres(split: str) -> dict[str, str]:
    """Return ``{chain: seqres}`` for a split from the vendored ATLAS CSV."""
    fname = "atlas_test.csv" if split == "test" else "atlas_val.csv"
    text = resources.files("premval.data").joinpath(fname).read_text(encoding="utf-8")
    return {row["name"]: row["seqres"] for row in csv.DictReader(text.splitlines())}


def _default_chains(split: str) -> list[str]:
    """Return the full chain list for a split."""
    return load_test_chains() if split == "test" else load_val_chains()


def _esmflow_command(
    repo: Path,
    ckpt: Path,
    input_csv: Path,
    out_dir: Path,
    n_samples: int,
) -> list[str]:
    """Build the ESMFlow distilled sampling subprocess command.

    Flags match the AlphaFlow README ESMFlow section plus the distilled
    addendum ("If running any distilled model, append the arguments
    --noisy_first --no_diffusion"). No ``--msa_dir`` because ESMFlow's
    ESMFold backbone is single-sequence.
    """
    return [
        sys.executable,
        str(repo / "predict.py"),
        "--mode",
        "esmfold",
        "--input_csv",
        str(input_csv),
        "--weights",
        str(ckpt),
        "--samples",
        str(n_samples),
        "--outpdb",
        str(out_dir),
        "--noisy_first",
        "--no_diffusion",
    ]


def _load_multi_model_pdb(path: Path, n_samples: int) -> md.Trajectory:
    """Load ESMFlow's per-chain multi-model PDB and validate the frame count."""
    import mdtraj as md

    from premval.io import enforce_ensemble_size

    if not path.exists():
        raise RuntimeError(f"ESMFlow produced no PDB at {path}")
    traj = md.load(str(path))
    return enforce_ensemble_size(traj, expected=n_samples)


def _sample_chain(
    chain: str,
    seq: str,
    ckpt: Path,
    repo: Path,
    n_samples: int,
) -> md.Trajectory:
    """Generate ``n_samples`` conformations for one chain via ESMFlow distilled.

    Heavy / GPU work lives entirely in the subprocess. Writes the chain as a
    one-row CSV (``name,seqres``), invokes ``predict.py``, then loads the
    resulting ``{chain}.pdb`` and returns it size-enforced.
    """
    with tempfile.TemporaryDirectory(prefix=f"esmflow_{chain}_") as tmp:
        tmp_path = Path(tmp)
        input_csv = tmp_path / "input.csv"
        with input_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "seqres"])
            writer.writerow([chain, seq])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cmd = _esmflow_command(repo, ckpt, input_csv, out_dir, n_samples)
        subprocess.run(cmd, check=True, cwd=str(repo))
        return _load_multi_model_pdb(out_dir / f"{chain}.pdb", n_samples)


def _checkpoint_path(checkpoint: str) -> Path:
    """Resolve the distilled checkpoint file path from the environment."""
    env = "ESMFLOW_MD_DISTILLED_CKPT" if checkpoint == "md" else "ESMFLOW_PDB_DISTILLED_CKPT"
    value = os.environ.get(env)
    if not value:
        raise SystemExit(f"set {env} to the ESMFlow {checkpoint} distilled checkpoint path")
    return Path(value)


def _repo_path() -> Path:
    """Resolve the AlphaFlow/ESMFlow repo clone path from the environment."""
    value = os.environ.get("ALPHAFLOW_REPO")
    if not value:
        raise SystemExit("set ALPHAFLOW_REPO to the AlphaFlow repo clone path")
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
    """Sample each chain with ESMFlow distilled and write a 250-MODEL PDB to the cache.

    Resumable: a chain whose output PDB already exists is skipped.
    """
    seqres = _seqres(split)
    repo = _repo_path()
    ckpt = _checkpoint_path(checkpoint)

    config = {"model": out_model, "split": split, "checkpoint": checkpoint, "n_samples": n_samples}
    with wandb_run(out_model, split, config) as logger:
        for chain in chains:
            dest = sample_path(out_model, chain, samples_dir)
            if dest.exists():
                print(f"skip {chain}: {dest} exists")
                continue
            seq = seqres.get(chain)
            if seq is None:
                raise SystemExit(f"no seqres for chain {chain!r} in split {split!r}")
            print(f"sample {chain} ({len(seq)} residues) -> {dest}")
            try:
                with track_sample(chain, n_samples) as sink:
                    traj = _sample_chain(chain, seq, ckpt, repo, n_samples)
            except subprocess.CalledProcessError as exc:
                # Most commonly a CUDA OOM in predict.py; log and move on so
                # one borderline-large chain doesn't kill the whole batch.
                print(f"FAILED {chain}: predict.py exited {exc.returncode}; continuing")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            traj.save_pdb(str(dest))
            telemetry = sink[0]
            write_telemetry(dest, telemetry)
            logger.log(telemetry)
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

    for ck, expected in (("pdb", "esmflow_pdb_distilled"), ("md", "esmflow_md_distilled")):
        assert DEFAULT_OUT_MODEL[ck] == expected, f"out-model default for {ck} should be {expected}"

    seqres = _seqres(split)
    chains = _default_chains(split)[:2]
    out_model = DEFAULT_OUT_MODEL["pdb"]

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
    parser = argparse.ArgumentParser(description="Run ESMFlow distilled on ATLAS chains.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--chains", nargs="+", default=None, help="Chain ids; default = whole split."
    )
    parser.add_argument("--checkpoint", choices=("pdb", "md"), default="pdb")
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
