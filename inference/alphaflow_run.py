"""Run AlphaFlow (distilled) to generate ATLAS conformational ensembles for PREMVAL.

AlphaFlow (https://github.com/bjing2016/alphaflow, MIT) is a flow-matching
generator that repurposes AlphaFold2 into a sequence-conditioned ensemble
sampler. This harness drives AlphaFlow once per ATLAS chain to produce a
250-MODEL multi-model PDB and drops it into PREMVAL's samples cache, where the
existing scorer picks it up.

The importable ``premval`` package is CPU-only and must never depend on
torch / GPU code. This script therefore lives OUTSIDE the package (top-level
``inference/``) and ``premval`` never imports it. All heavy imports (torch,
mdtraj, the AlphaFlow repo) are performed lazily INSIDE the sampling function,
so this module imports and ``--self-test`` runs with only the CPU ``premval``
install present.

Checkpoint variants and contamination implications
---------------------------------------------------
AlphaFlow publishes weights on HuggingFace (``bjing-mit/alphaflow``). This
harness defaults to the **distilled** variants (much cheaper than the base
diffusion checkpoints, comparable quality per the paper). Two are exposed
via ``--checkpoint``:

- ``pdb``: PDB-trained distilled. Never saw ATLAS MD trajectories, so
  ATLAS evaluation is held-out. Default out-model ``alphaflow_pdb_distilled``;
  contamination label ``test``.
- ``md``: ATLAS-fine-tuned distilled. Trained on ATLAS MD, so scoring it on
  the ATLAS val/test splits is in-distribution. Default out-model
  ``alphaflow_md_distilled``; contamination label ``in_distribution``.

The contamination labels are recorded in
``data/contamination_labels.yaml`` for the operator's bookkeeping; this
script only writes the ensemble PDBs.

AlphaFlow setup (performed manually on a GPU host)
--------------------------------------------------
1. Clone the repo and create its environment::

       git clone https://github.com/bjing2016/alphaflow
       cd alphaflow
       conda env create -f environment.yml      # or follow the repo README
       conda activate alphaflow

2. Download the distilled weights from HuggingFace ``bjing-mit/alphaflow``,
   e.g.::

       huggingface-cli download bjing-mit/alphaflow \\
           alphaflow_pdb_distilled_202402.pt alphaflow_md_distilled_202402.pt \\
           --local-dir ./weights

3. Provide MSAs at ``{msa_dir}/{chain}/a3m/{chain}.a3m`` (AlphaFlow needs
   them; the ESMFlow harness does not). The AlphaFlow repo's
   ``splits/atlas_test.csv`` lists the same chain ids PREMVAL uses, and the
   repo's ``scripts/mmseqs_query.py`` can populate this directory from the
   ColabFold server in one shot.

4. Point this script at the clone, weights, and MSAs via environment
   variables::

       export ALPHAFLOW_REPO=/path/to/alphaflow
       export ALPHAFLOW_PDB_DISTILLED_CKPT=/path/to/weights/alphaflow_pdb_distilled_202402.pt
       export ALPHAFLOW_MD_DISTILLED_CKPT=/path/to/weights/alphaflow_md_distilled_202402.pt
       export ALPHAFLOW_MSA_DIR=/path/to/msa_dir

Inference invocation
--------------------
This harness shells out to the repo's ``predict.py`` rather than importing
internals; ``_alphaflow_command`` assembles the documented flags
(``--mode alphafold --noisy_first --no_diffusion --samples 250``, per the
AlphaFlow README AlphaFlow + distilled sections). ``predict.py`` writes one
multi-model PDB per input row (``{outpdb}/{name}.pdb``); this harness loads
it and re-emits at ``sample_path(out_model, chain, samples_dir)`` after
verifying the frame count.

Run command
-----------
::

    python inference/alphaflow_run.py --split val  --checkpoint pdb
    python inference/alphaflow_run.py --split test --checkpoint md
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/alphaflow_run.py --self-test
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
    "pdb": "alphaflow_pdb_distilled",
    "md": "alphaflow_md_distilled",
}


def _seqres(split: str) -> dict[str, str]:
    """Return ``{chain: seqres}`` for a split from the vendored ATLAS CSV."""
    fname = "atlas_test.csv" if split == "test" else "atlas_val.csv"
    text = resources.files("premval.data").joinpath(fname).read_text(encoding="utf-8")
    return {row["name"]: row["seqres"] for row in csv.DictReader(text.splitlines())}


def _default_chains(split: str) -> list[str]:
    """Return the full chain list for a split."""
    return load_test_chains() if split == "test" else load_val_chains()


def _alphaflow_command(
    repo: Path,
    ckpt: Path,
    input_csv: Path,
    msa_dir: Path,
    out_dir: Path,
    n_samples: int,
) -> list[str]:
    """Build the AlphaFlow distilled sampling subprocess command.

    Flags match the AlphaFlow README ("python predict.py --mode alphafold
    --input_csv ... --msa_dir ... --weights ... --samples N --outpdb ...")
    plus the distilled addendum ("If running any distilled model, append the
    arguments --noisy_first --no_diffusion"). The ``--self_cond --resample``
    suggestion in the README applies to the PDB-base diffusion model, not
    the distilled variants.
    """
    return [
        sys.executable,
        str(repo / "predict.py"),
        "--mode",
        "alphafold",
        "--input_csv",
        str(input_csv),
        "--msa_dir",
        str(msa_dir),
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
    """Load AlphaFlow's per-chain multi-model PDB and validate the frame count."""
    import mdtraj as md

    from premval.io import enforce_ensemble_size

    if not path.exists():
        raise RuntimeError(f"AlphaFlow produced no PDB at {path}")
    traj = md.load(str(path))
    return enforce_ensemble_size(traj, expected=n_samples)


def _sample_chain(
    chain: str,
    seq: str,
    ckpt: Path,
    repo: Path,
    msa_dir: Path,
    n_samples: int,
) -> md.Trajectory:
    """Generate ``n_samples`` conformations for one chain via AlphaFlow distilled.

    Heavy / GPU work lives entirely in the subprocess. Writes the chain as a
    one-row CSV (``name,seqres``), invokes ``predict.py``, then loads the
    resulting ``{chain}.pdb`` and returns it size-enforced.
    """
    with tempfile.TemporaryDirectory(prefix=f"alphaflow_{chain}_") as tmp:
        tmp_path = Path(tmp)
        input_csv = tmp_path / "input.csv"
        with input_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "seqres"])
            writer.writerow([chain, seq])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cmd = _alphaflow_command(repo, ckpt, input_csv, msa_dir, out_dir, n_samples)
        subprocess.run(cmd, check=True, cwd=str(repo))
        return _load_multi_model_pdb(out_dir / f"{chain}.pdb", n_samples)


def _checkpoint_path(checkpoint: str) -> Path:
    """Resolve the distilled checkpoint file path from the environment."""
    env = "ALPHAFLOW_MD_DISTILLED_CKPT" if checkpoint == "md" else "ALPHAFLOW_PDB_DISTILLED_CKPT"
    value = os.environ.get(env)
    if not value:
        raise SystemExit(f"set {env} to the AlphaFlow {checkpoint} distilled checkpoint path")
    return Path(value)


def _repo_path() -> Path:
    """Resolve the AlphaFlow repo clone path from the environment."""
    value = os.environ.get("ALPHAFLOW_REPO")
    if not value:
        raise SystemExit("set ALPHAFLOW_REPO to the AlphaFlow repo clone path")
    return Path(value)


def _msa_dir() -> Path:
    """Resolve the per-chain MSA directory from the environment."""
    value = os.environ.get("ALPHAFLOW_MSA_DIR")
    if not value:
        raise SystemExit("set ALPHAFLOW_MSA_DIR to the directory of {chain}.a3m files")
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
    """Sample each chain with AlphaFlow distilled and write a 250-MODEL PDB to the cache.

    Resumable: a chain whose output PDB already exists is skipped.
    """
    seqres = _seqres(split)
    repo = _repo_path()
    ckpt = _checkpoint_path(checkpoint)
    msa_dir = _msa_dir()

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
            with track_sample(chain, n_samples) as sink:
                traj = _sample_chain(chain, seq, ckpt, repo, msa_dir, n_samples)
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

    for ck, expected in (("pdb", "alphaflow_pdb_distilled"), ("md", "alphaflow_md_distilled")):
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
    parser = argparse.ArgumentParser(description="Run AlphaFlow distilled on ATLAS chains.")
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
