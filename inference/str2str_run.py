"""Str2Str harness: generate per-chain conformational ensembles for PREMVAL.

Wraps `Str2Str` (https://github.com/lujiarui/Str2Str, MIT-licensed), a
zero-shot conformation sampler. Str2Str takes a single input structure
(here, each ATLAS chain's crystal/topology PDB), perturbs it with a forward
SDE, and denoises back to emit a new conformation. Running it N times yields
an ensemble. This script drives Str2Str over the ATLAS val/test chains,
collects 250 conformations per chain, and writes each as a 250-MODEL
multi-model PDB into PREMVAL's samples cache so the existing scorer can rate
the `str2str` model alongside the published baselines.

CPU-only-package constraint
----------------------------
The importable `premval` package stays CPU-only and GPU-dependency-free. All
heavy/GPU machinery (torch, PyTorch Lightning, Hydra, the Str2Str repo) is
confined to this `inference/` directory, which `premval` never imports. To
honor that, every heavy import in this module is **lazy** (inside the
sampling function), so the module imports and `--self-test` runs with only
the CPU `premval` install present.

Str2Str setup (run manually before the real GPU run)
----------------------------------------------------
1. Clone the repo somewhere outside this package::

       git clone https://github.com/lujiarui/Str2Str.git
       cd Str2Str

2. Create the conda/pip env per the repo's README (PyTorch + Lightning +
   Hydra stack), e.g.::

       conda env create -f environment.yaml
       conda activate str2str
       pip install -e .

3. Download the pretrained checkpoint from the Google Drive link in the
   Str2Str README into the repo's `data/ckpt` directory (the path the
   inference config expects).

4. Point this harness at the clone with ``--str2str-repo /path/to/Str2Str``
   (or the ``STR2STR_REPO`` env var).

How this script invokes Str2Str
--------------------------------
Str2Str ships a Hydra/Lightning inference entrypoint (`scripts/eval.py`,
invoked through its own console entry) that reads structures from an input
directory and writes sampled PDBs to an output directory. This harness, for
each chain:

  1. writes the chain's topology PDB (from `load_topology_bytes`) into a temp
     input dir,
  2. shells out to the Str2Str inference CLI requesting `--n-samples`
     conformations with the SDE sampler,
  3. loads the produced conformations with mdtraj, stacks exactly
     `--n-samples` frames onto the chain topology, and saves them as a single
     multi-model PDB at the PREMVAL `sample_path`.

The exact Str2Str CLI flags vary by repo revision; the command assembled in
`_run_str2str` is the documented assumption. The GPU run is executed manually
by the user, who will adjust `_run_str2str` if their checkout's invocation
differs. Everything else (chain selection, output path, frame count, the
multi-model PDB write) is fixed by the PREMVAL shared harness contract.

Run command
-----------
    # validate plumbing without a GPU / torch / Str2Str:
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/str2str_run.py --self-test

    # real run on the val split (needs a GPU + Str2Str installed):
    python inference/str2str_run.py --split val --str2str-repo /path/to/Str2Str
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mdtraj as md

from premval.data import (
    default_samples_dir,
    load_test_chains,
    load_topology_bytes,
    load_val_chains,
    sample_path,
)

DEFAULT_OUT_MODEL = "str2str"
DEFAULT_N_SAMPLES = 250


def _topology_to_tempfile(chain: str, directory: Path) -> Path:
    """Write a chain's crystal/topology PDB bytes to `{directory}/{chain}.pdb`."""
    pdb_path = directory / f"{chain}.pdb"
    pdb_path.write_bytes(load_topology_bytes(chain))
    return pdb_path


def _run_str2str(input_dir: Path, output_dir: Path, n_samples: int, repo: Path) -> None:
    """Shell out to the Str2Str inference CLI to sample conformations.

    Drives Str2Str's Hydra/Lightning inference entrypoint over every PDB in
    `input_dir`, requesting `n_samples` SDE-sampled conformations per input and
    writing the results under `output_dir`. The flag layout reflects the
    documented assumption (see module docstring); adjust here if the local
    checkout's CLI differs.

    Args:
        input_dir: Directory holding the chain topology PDB(s) to sample from.
        output_dir: Directory Str2Str writes its sampled PDB(s) into.
        n_samples: Number of conformations to draw per input structure.
        repo: Path to the local Str2Str clone (the `scripts/eval.py` entry).

    Raises:
        FileNotFoundError: If `repo` or its inference entrypoint is missing.
        subprocess.CalledProcessError: If the Str2Str run exits non-zero.
    """
    entrypoint = repo / "scripts" / "eval.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"Str2Str inference entrypoint not found at {entrypoint}; "
            "clone https://github.com/lujiarui/Str2Str and pass --str2str-repo"
        )
    command = [
        sys.executable,
        str(entrypoint),
        f"input_dir={input_dir}",
        f"output_dir={output_dir}",
        "sampling.sde=true",
        f"sampling.n_samples={n_samples}",
    ]
    subprocess.run(command, cwd=repo, check=True)


def _collect_ensemble(topology_pdb: Path, output_dir: Path, n_samples: int) -> md.Trajectory:
    """Stack Str2Str's sampled conformations into one trajectory of n_samples frames.

    Loads every PDB Str2Str emitted under `output_dir`, concatenates their
    frames onto the chain topology, and slices to exactly `n_samples`.

    Args:
        topology_pdb: The chain topology PDB fed to Str2Str (defines atoms).
        output_dir: Directory Str2Str wrote its sampled conformations into.
        n_samples: Required number of frames in the returned trajectory.

    Returns:
        An `mdtraj.Trajectory` with exactly `n_samples` frames.

    Raises:
        ValueError: If fewer than `n_samples` conformations were produced.
    """
    import mdtraj as md

    sampled = sorted(output_dir.rglob("*.pdb"))
    if not sampled:
        raise ValueError(f"Str2Str produced no PDBs under {output_dir}")
    traj = md.join([md.load(str(p), top=str(topology_pdb)) for p in sampled])
    if traj.n_frames < n_samples:
        raise ValueError(f"Str2Str produced {traj.n_frames} conformations; need {n_samples}")
    return traj[:n_samples]


def sample_chain(
    chain: str,
    dest: Path,
    n_samples: int,
    repo: Path,
) -> None:
    """Sample one chain with Str2Str and write the ensemble to `dest`.

    All heavy imports (mdtraj, torch via the subprocess) are confined here so
    the module stays importable on a CPU-only premval install.

    Args:
        chain: PDB chain identifier, e.g. `6o2v_A`.
        dest: Output multi-model PDB path (created, parent must exist).
        n_samples: Number of conformations to generate.
        repo: Path to the local Str2Str clone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        topology_pdb = _topology_to_tempfile(chain, input_dir)
        _run_str2str(input_dir, output_dir, n_samples, repo)
        traj = _collect_ensemble(topology_pdb, output_dir, n_samples)
        traj.save_pdb(str(dest))


def _self_test(chains: list[str], n_samples: int, samples_dir: Path) -> None:
    """Exercise the topology read and output path with synthetic frames (no GPU).

    For the first one or two chains: read the topology via `load_topology_bytes`,
    build a synthetic trajectory of exactly `n_samples` random frames on that
    topology, write it through the real `sample_path`, reload it with
    `premval.io.load_ensemble`, and assert the frame count and location.
    """
    import mdtraj as md
    import numpy as np

    from premval.io import load_ensemble

    rng = np.random.default_rng(0)
    for chain in chains[:2]:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
            tmp.write(load_topology_bytes(chain))
            tmp.flush()
            top_traj = md.load(tmp.name)
        base = top_traj.xyz[0]
        xyz = base[None] + rng.normal(scale=0.05, size=(n_samples,) + base.shape)
        synthetic = md.Trajectory(xyz.astype("float32"), top_traj.topology)

        dest = sample_path(DEFAULT_OUT_MODEL, chain, samples_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        synthetic.save_pdb(str(dest))

        assert dest == sample_path(DEFAULT_OUT_MODEL, chain, samples_dir)
        assert dest.exists(), f"{chain}: nothing written at {dest}"
        reloaded = load_ensemble(dest)
        assert reloaded.n_frames == n_samples, (
            f"{chain}: wrote {n_samples} frames but reloaded {reloaded.n_frames}"
        )
        print(f"PASS self-test {chain}: {reloaded.n_frames} frames at {dest}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--chains", nargs="+", default=None, help="Override the split chains.")
    parser.add_argument("--out-model", default=DEFAULT_OUT_MODEL)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Samples cache root (defaults to PREMVAL's default_samples_dir).",
    )
    parser.add_argument(
        "--str2str-repo",
        type=Path,
        default=os.environ.get("STR2STR_REPO"),
        help="Path to the local Str2Str clone (or set STR2STR_REPO).",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    chains = args.chains or (load_val_chains() if args.split == "val" else load_test_chains())
    samples_dir = args.samples_dir or default_samples_dir()

    if args.self_test:
        _self_test(chains, args.n_samples, samples_dir)
        return 0

    if args.str2str_repo is None:
        raise SystemExit("--str2str-repo (or STR2STR_REPO) is required for a real run")

    for chain in chains:
        dest = sample_path(args.out_model, chain, samples_dir)
        if dest.exists():
            print(f"skip {chain}: already at {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"sampling {chain} -> {dest}")
        sample_chain(chain, dest, args.n_samples, args.str2str_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
