"""ESMDiff harness: generate per-chain conformational ensembles for PREMVAL.

Wraps `ESMDiff` (https://github.com/lujiarui/esmdiff), the structure-language
model from Lu et al., "Structure Language Models for Protein Conformation
Generation" (ICLR 2025). ESMDiff fine-tunes ESM3 with masked (discrete)
diffusion over structure tokens, then samples those tokens and decodes them back
to 3D. This harness drives ESMDiff over the ATLAS val/test chains, collects 250
conformations per chain, and writes each as a 250-MODEL multi-model PDB into
PREMVAL's samples cache so the existing scorer can rate the `esmdiff` model
alongside the published baselines.

Sequence-conditioned, fed from the vendored sequences
-----------------------------------------------------
ESMDiff's `slm/sample_esmdiff.py` only accepts `--input` as a *directory of PDB
files*: it reads each PDB and extracts the amino-acid sequence (and, only in
inpainting mode, the coordinates) via `ESMProtein.from_pdb`. In the default
sampling path (no `--mask_ids`) the structure tokens are sampled from scratch and
the input coordinates are **not** used, so generation is conditioned purely on
sequence. The PDB is therefore just a sequence carrier, not a structure input.

We get each chain's sequence the cheap way every other sequence-conditioned
harness does (`load_chain_sequences`, straight from the vendored ATLAS CSV) and
write a throwaway placeholder PDB whose residue names spell that sequence; its
idealized backbone coordinates are discarded by ESMDiff. This keeps the harness
free of the ATLAS trajectory bundles (no download to read a sequence) and avoids
feeding ESMDiff the reference structure, so it stays comparable to the other
sequence-only generators on the leaderboard. (Inpainting mode, which *would*
condition on coordinates, is deliberately not used here.)

CPU-only-package constraint
----------------------------
The importable `premval` package stays CPU-only and GPU-dependency-free. All
heavy/GPU machinery (torch, ESM3, the ESMDiff repo) is confined to this
`inference/` directory, which `premval` never imports. Every heavy import in
this module is **lazy** (inside the sampling function), so the module imports and
`--self-test` runs with only the CPU `premval` install present.

ESMDiff setup (run manually before the real GPU run)
----------------------------------------------------
1. Clone the repo somewhere outside this package::

       git clone https://github.com/lujiarui/esmdiff.git

2. Create the env per the repo's README (Python 3.10, PyTorch + Lightning +
   Hydra), then install the gated ESM3 weights: accept the license at
   https://huggingface.co/EvolutionaryScale/esm3 and `huggingface-cli login`
   (ESMDiff downloads `esm3_sm_open_v1` on first use).

3. Download the ESMDiff checkpoint (`release_v0.pt`) from the Google Drive link
   in the README into the repo's `data/ckpt/` directory.

4. Point this harness at the clone with ``--esmdiff-repo /path/to/esmdiff``
   (or the ``ESMDIFF_REPO`` env var). The checkpoint defaults to
   ``{repo}/data/ckpt/release_v0.pt``; override with ``--ckpt``.

How this script invokes ESMDiff
-------------------------------
For each chain this harness:

  1. writes a placeholder PDB spelling the chain's sequence into a temp input
     dir (ESMDiff reads only the sequence from it),
  2. shells out to `slm/sample_esmdiff.py` requesting `--n-samples`
     conformations,
  3. loads ESMDiff's merged multi-model PDB, enforces exactly `--n-samples`
     frames (the leaderboard contract; see `premval.io.enforce_ensemble_size`),
     and writes it to the PREMVAL `sample_path`.

The `--mode`/`--num-steps` defaults match ESMDiff's published flagship
configuration; the exact CLI may vary by repo revision, so adjust the command in
`_run_esmdiff` if your checkout differs (same contract as `str2str_run`).

Run command
-----------
    # validate plumbing without a GPU / torch / ESMDiff:
    PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/esmdiff_run.py --self-test

    # real run on the val split (needs a GPU + ESMDiff installed):
    python inference/esmdiff_run.py --split val --esmdiff-repo /path/to/esmdiff
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mdtraj as md

from common import track_sample, wandb_run, write_telemetry

from premval.data import (
    default_samples_dir,
    load_chain_sequences,
    load_test_chains,
    load_val_chains,
    sample_path,
)

DEFAULT_OUT_MODEL = "esmdiff"
DEFAULT_N_SAMPLES = 250
# ESMDiff's discrete-diffusion sampler ("ddpm") is the configuration whose
# ensemble metrics the paper reports as the flagship result; "gibbs" is the
# alternative. 25 steps is the repo's documented default.
DEFAULT_MODE = "ddpm"
DEFAULT_NUM_STEPS = 25

# One-letter -> three-letter residue names, so a synthesized PDB's ATOM records
# spell the chain sequence that ESMDiff's `ESMProtein.from_pdb` reads back.
_THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip


def _write_sequence_pdb(chain: str, sequence: str, directory: Path) -> Path:
    """Write a placeholder backbone PDB spelling `sequence` to `{directory}/{chain}.pdb`.

    ESMDiff's CLI only ingests PDBs, but in default mode it uses only the parsed
    sequence (the coordinates are discarded). So we emit one residue per
    character with idealized, non-degenerate N/CA/C/O backbone coordinates: the
    residue *names* carry the sequence ESMDiff reads back, and the geometry is
    irrelevant. This lets the harness drive ESMDiff from the vendored sequence
    alone, with no ATLAS structure download.

    Args:
        chain: Chain identifier, used as the file stem (ESMDiff names its output
            after it).
        sequence: One-letter amino-acid sequence (standard 20 residues).
        directory: Directory to write `{chain}.pdb` into.

    Returns:
        Path to the written PDB.

    Raises:
        KeyError: If `sequence` contains a non-standard one-letter code.
    """
    lines: list[str] = []
    serial = 1
    for i, aa in enumerate(sequence):
        resname = _THREE_LETTER[aa]
        base = 3.8 * i
        # Non-collinear N-CA-C so the backbone frame is well-defined; ~3.8 A
        # CA-CA spacing keeps consecutive residues apart.
        atoms = (
            ("N", base + 0.000, 0.000, 0.000),
            ("CA", base + 1.458, 0.000, 0.000),
            ("C", base + 2.000, 1.400, 0.000),
            ("O", base + 1.700, 2.550, 0.000),
        )
        for name, x, y, z in atoms:
            # Fixed-column PDB ATOM record: name 13-16, altLoc 17, resName 18-20,
            # chainID 22, resSeq 23-26, x/y/z 31-54, occ/temp 55-66, element 77-78.
            lines.append(
                f"ATOM  {serial:>5d} {name:^4s} {resname:>3s} A{i + 1:>4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}"
            )
            serial += 1
    lines.append("END")
    pdb_path = directory / f"{chain}.pdb"
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pdb_path


def _run_esmdiff(
    input_dir: Path,
    output_dir: Path,
    n_samples: int,
    repo: Path,
    ckpt: Path,
    *,
    mode: str,
    num_steps: int,
) -> None:
    """Shell out to ESMDiff's `slm/sample_esmdiff.py` to sample conformations.

    Drives ESMDiff over every PDB in `input_dir`, requesting `n_samples`
    conformations per input (sequence-only, default mode: no `--mask_ids`) and
    writing the merged multi-model PDB(s) under `output_dir`.

    Args:
        input_dir: Directory holding the placeholder sequence PDB(s).
        output_dir: Directory ESMDiff writes its sampled PDB(s) into.
        n_samples: Number of conformations to draw per input structure.
        repo: Path to the local ESMDiff clone (the `slm/sample_esmdiff.py` entry).
        ckpt: Path to the ESMDiff checkpoint (`release_v0.pt`).
        mode: ESMDiff sampler, `ddpm` or `gibbs`.
        num_steps: Number of diffusion steps.

    Raises:
        FileNotFoundError: If `repo`'s entrypoint or `ckpt` is missing.
        subprocess.CalledProcessError: If the ESMDiff run exits non-zero.
    """
    entrypoint = repo / "slm" / "sample_esmdiff.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"ESMDiff inference entrypoint not found at {entrypoint}; "
            "clone https://github.com/lujiarui/esmdiff and pass --esmdiff-repo"
        )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"ESMDiff checkpoint not found at {ckpt}; download release_v0.pt from "
            "the README's Google Drive link into the repo's data/ckpt/ (or pass --ckpt)"
        )
    command = [
        sys.executable,
        str(entrypoint),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--ckpt",
        str(ckpt),
        "--mode",
        mode,
        "--num_samples",
        str(n_samples),
        "--num_steps",
        str(num_steps),
    ]
    subprocess.run(command, cwd=repo, check=True)


def _collect_ensemble(chain: str, output_dir: Path, n_samples: int) -> md.Trajectory:
    """Load ESMDiff's merged multi-model PDB and enforce exactly n_samples frames.

    ESMDiff writes one merged multi-model PDB per input, named after the input
    stem (`{chain}.pdb`), inside a timestamped subdirectory of `output_dir`. The
    per-sample shards it merges are named `{chain}.{i}.pdb`, so globbing the
    exact `{chain}.pdb` stem selects the merged file without double-counting.

    Args:
        chain: Chain identifier (the input PDB stem ESMDiff names its output by).
        output_dir: Directory ESMDiff wrote its results into.
        n_samples: Required number of frames in the returned trajectory.

    Returns:
        An `mdtraj.Trajectory` with exactly `n_samples` frames.

    Raises:
        FileNotFoundError: If no merged `{chain}.pdb` was produced.
        ValueError: If fewer than `n_samples` conformations were produced.
    """
    import mdtraj as md

    from premval.io import enforce_ensemble_size

    merged = sorted(output_dir.rglob(f"{chain}.pdb"))
    if not merged:
        raise FileNotFoundError(
            f"ESMDiff produced no merged {chain}.pdb under {output_dir}; "
            "check the repo revision's output layout (see _run_esmdiff)"
        )
    traj = md.load(str(merged[0]))
    if traj.n_frames < n_samples:
        raise ValueError(
            f"ESMDiff produced {traj.n_frames} conformations for {chain}; need {n_samples}. "
            "Re-run with a larger --n-samples."
        )
    return enforce_ensemble_size(traj, expected=n_samples)


def sample_chain(
    chain: str,
    sequence: str,
    dest: Path,
    n_samples: int,
    repo: Path,
    ckpt: Path,
    *,
    mode: str,
    num_steps: int,
) -> None:
    """Sample one chain with ESMDiff and write the ensemble to `dest`.

    All heavy imports (mdtraj, torch/ESM3 via the subprocess) are confined here
    so the module stays importable on a CPU-only premval install.

    Args:
        chain: PDB chain identifier, e.g. `6o2v_A`.
        sequence: One-letter amino-acid sequence to condition on.
        dest: Output multi-model PDB path (created, parent must exist).
        n_samples: Number of conformations to generate.
        repo: Path to the local ESMDiff clone.
        ckpt: Path to the ESMDiff checkpoint.
        mode: ESMDiff sampler, `ddpm` or `gibbs`.
        num_steps: Number of diffusion steps.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        _write_sequence_pdb(chain, sequence, input_dir)
        _run_esmdiff(input_dir, output_dir, n_samples, repo, ckpt, mode=mode, num_steps=num_steps)
        traj = _collect_ensemble(chain, output_dir, n_samples)
        traj.save_pdb(str(dest))


def _self_test(split: str, n_samples: int, samples_dir: Path) -> None:
    """Exercise the sequence->PDB write and output path with synthetic frames (no GPU).

    For the first one or two chains: build the placeholder sequence PDB, confirm
    mdtraj loads it and its residue count matches the sequence, then write a
    synthetic trajectory of exactly `n_samples` frames on that topology through
    the real `sample_path`, reload it with `premval.io.load_ensemble`, and assert
    the frame count and location.
    """
    import mdtraj as md
    import numpy as np

    from premval.io import load_ensemble

    rng = np.random.default_rng(0)
    sequences = load_chain_sequences(split)
    for chain in list(sequences)[:2]:
        sequence = sequences[chain]
        with tempfile.TemporaryDirectory() as tmp:
            pdb = _write_sequence_pdb(chain, sequence, Path(tmp))
            top_traj = md.load(str(pdb))
        assert top_traj.n_residues == len(sequence), (
            f"{chain}: placeholder PDB has {top_traj.n_residues} residues, seq has {len(sequence)}"
        )
        base = top_traj.xyz[0]
        xyz = base[None] + rng.normal(scale=0.05, size=(n_samples,) + base.shape)
        synthetic = md.Trajectory(xyz.astype("float32"), top_traj.topology)

        dest = sample_path(DEFAULT_OUT_MODEL, chain, samples_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with track_sample(chain, n_samples) as sink:
            synthetic.save_pdb(str(dest))
        telemetry = sink[0]
        sidecar = write_telemetry(dest, telemetry)

        assert dest.exists(), f"{chain}: nothing written at {dest}"
        reloaded = load_ensemble(dest)
        assert reloaded.n_frames == n_samples, (
            f"{chain}: wrote {n_samples} frames but reloaded {reloaded.n_frames}"
        )
        assert sidecar.exists(), f"{chain}: no telemetry sidecar at {sidecar}"
        recorded = json.loads(sidecar.read_text())
        assert recorded["chain"] == chain and recorded["wall_seconds"] >= 0.0
        print(f"PASS self-test {chain}: {reloaded.n_frames} frames at {dest}")
        print(f"  {telemetry.summary()}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--chains", nargs="+", default=None, help="Override the split chains.")
    parser.add_argument("--out-model", default=DEFAULT_OUT_MODEL)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--mode", choices=("ddpm", "gibbs"), default=DEFAULT_MODE)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Samples cache root (defaults to PREMVAL's default_samples_dir).",
    )
    parser.add_argument(
        "--esmdiff-repo",
        type=Path,
        default=os.environ.get("ESMDIFF_REPO"),
        help="Path to the local ESMDiff clone (or set ESMDIFF_REPO).",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="ESMDiff checkpoint (defaults to {repo}/data/ckpt/release_v0.pt).",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples_dir = args.samples_dir or default_samples_dir()

    if args.self_test:
        _self_test(args.split, args.n_samples, samples_dir)
        return 0

    if args.esmdiff_repo is None:
        raise SystemExit("--esmdiff-repo (or ESMDIFF_REPO) is required for a real run")
    ckpt = args.ckpt or (args.esmdiff_repo / "data" / "ckpt" / "release_v0.pt")

    sequences = load_chain_sequences(args.split)
    chains = args.chains or (load_val_chains() if args.split == "val" else load_test_chains())

    config = {
        "model": args.out_model,
        "split": args.split,
        "n_samples": args.n_samples,
        "mode": args.mode,
        "num_steps": args.num_steps,
    }
    with wandb_run(args.out_model, args.split, config) as logger:
        for chain in chains:
            dest = sample_path(args.out_model, chain, samples_dir)
            if dest.exists():
                print(f"skip {chain}: already at {dest}")
                continue
            if chain not in sequences:
                raise KeyError(f"chain {chain!r} not in {args.split} split CSV")
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"sampling {chain} -> {dest}")
            with track_sample(chain, args.n_samples) as sink:
                sample_chain(
                    chain,
                    sequences[chain],
                    dest,
                    args.n_samples,
                    args.esmdiff_repo,
                    ckpt,
                    mode=args.mode,
                    num_steps=args.num_steps,
                )
            telemetry = sink[0]
            write_telemetry(dest, telemetry)
            logger.log(telemetry)
            print(telemetry.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
