"""Modal wrapper around ``inference/esmflow_run.py``.

Builds a CUDA-11.8 image with the AlphaFlow repo + ESMFlow distilled weights,
mounts the local PREMVAL checkout, and runs ``esmflow_run.py`` against the
ATLAS val/test splits on an A100-80GB. Generated samples land on a Modal
Volume (``premval-samples``) under ``samples/{out_model}/{chain}.pdb`` plus
the matching ``{chain}.telemetry.json`` sidecar, mirroring the local cache
layout so ``modal volume get`` can sync them straight into
``~/.cache/premval/samples/``.

``premval`` itself is never imported on Modal: the script is invoked as a
subprocess (same contract as a local GPU run), so this file is the only
GPU-aware code path on the wrapper side.

Launch
------
::

    set -a; source .env; set +a   # MODAL_TOKEN_ID/SECRET
    modal run inference/esmflow_modal.py             # both checkpoints, 5 chains
    modal run inference/esmflow_modal.py --checkpoints pdb --n-chains 5
    modal run inference/esmflow_modal.py --split test --checkpoints md

After the run, pull the samples back locally::

    modal volume get premval-samples samples ~/.cache/premval/

The volume only ever grows; re-launches skip chains whose PDB already exists.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHAFLOW_GIT = "https://github.com/bjing2016/alphaflow"
HF_WEIGHTS = "https://huggingface.co/bjing-mit/alphaflow/resolve/main/params"

# Per-checkpoint distilled weight filenames (the bjing-mit/alphaflow HF naming).
WEIGHTS: dict[str, str] = {
    "pdb": "esmflow_pdb_distilled_202402.pt",
    "md": "esmflow_md_distilled_202402.pt",
}

# Pinned by the alphaflow README's Installation section: torch 1.12.1+cu113
# and the deps below. Python is 3.10 here (Modal no longer ships 3.9, and the
# pinned numpy/torch wheels have cp310 builds). CUDA 11.8 devel satisfies
# openfold's "CUDA 11" requirement.
# The alphaflow README pins were locked for Python 3.9 era; Modal only ships
# 3.10+, so numpy/scipy are bumped to the closest co-compatible pair that
# still works with torch 1.12.1 (which was built against numpy < 1.24). The
# rest match the README.
_PIP_DEPS = [
    "numpy==1.23.5",
    "scipy==1.10.1",
    "pandas==1.5.3",
    "biopython==1.79",
    "dm-tree==0.1.6",
    "modelcif==0.7",
    "ml-collections==0.1.0",
    "absl-py",
    "einops",
    "pytorch_lightning==2.0.4",
    "fair-esm",
    "mdtraj==1.10.0",  # README pins 1.9.9 (no cp310 wheel); 1.10.0 is the first with one
    "wandb",
    # premval itself isn't pip-installed (its deps conflict with the alphaflow
    # stack on numpy/scipy/mdtraj), but the harness loads `premval.data` which
    # transitively imports sklearn.decomposition.PCA from premval.data.references.
    "scikit-learn==1.3.2",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "wget", "build-essential", "clang")
    .pip_install(
        "torch==1.12.1+cu113",
        extra_index_url="https://download.pytorch.org/whl/cu113",
    )
    .pip_install(*_PIP_DEPS)
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        # The pip_install above pulls torchmetrics (via pytorch_lightning),
        # which upgrades torch to a 2.x / CUDA 13 build. Force it back to
        # the torch 1.12.1+cu113 that openfold and the alphaflow README
        # expect; --no-deps keeps the rest of the env untouched.
        "pip install --force-reinstall --no-deps "
        "torch==1.12.1+cu113 "
        "--extra-index-url https://download.pytorch.org/whl/cu113",
        # openfold has CUDA extensions that compile against CUDA_HOME.
        # --no-build-isolation makes the build use our installed torch
        # rather than pip's isolated env (which would pull a newer torch).
        "pip install --no-build-isolation "
        "'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037'",
        f"git clone {ALPHAFLOW_GIT} /opt/alphaflow",
        # predict.py was updated for torch 2.x (line 76:
        # `torch.load(..., weights_only=False)`). torch 1.12 doesn't accept
        # that kwarg but its default behavior matches, so strip it.
        "sed -i 's/, weights_only=False//' /opt/alphaflow/predict.py",
        "mkdir -p /opt/weights",
        f"wget -q -O /opt/weights/{WEIGHTS['pdb']} {HF_WEIGHTS}/{WEIGHTS['pdb']}",
        f"wget -q -O /opt/weights/{WEIGHTS['md']} {HF_WEIGHTS}/{WEIGHTS['md']}",
    )
    .add_local_dir(
        local_path=str(REPO_ROOT),
        remote_path="/opt/premval",
        ignore=[
            "**/.venv/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
            "**/.coverage",
            "**/.env",
            "**/htmlcov/**",
            "**/node_modules/**",
        ],
    )
)

app = modal.App("premval-esmflow-distilled")
samples_volume = modal.Volume.from_name("premval-samples", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",  # default; per-invocation override via `sample.with_options(gpu=...)`
    volumes={"/cache": samples_volume},
    timeout=60 * 60 * 6,
)
def sample(checkpoint: str, split: str, chains: list[str]) -> None:
    """Run ``esmflow_run.py`` for one checkpoint over a list of chains.

    The Modal Volume is committed at exit so partial progress survives a
    crash, and the harness's skip-if-PDB-exists rule keeps re-runs idempotent.
    """
    import os
    import subprocess

    env = dict(os.environ)
    env.update(
        {
            "ALPHAFLOW_REPO": "/opt/alphaflow",
            "ESMFLOW_PDB_DISTILLED_CKPT": f"/opt/weights/{WEIGHTS['pdb']}",
            "ESMFLOW_MD_DISTILLED_CKPT": f"/opt/weights/{WEIGHTS['md']}",
            "PREMVAL_SAMPLES_DIR": "/cache/samples",
            "PYTHONPATH": "/opt/premval/src",
        }
    )
    cmd = [
        "python",
        "/opt/premval/inference/esmflow_run.py",
        "--split",
        split,
        "--checkpoint",
        checkpoint,
        "--chains",
        *chains,
    ]
    print("running:", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        samples_volume.commit()


@app.local_entrypoint()
def main(
    split: str = "val",
    checkpoints: str = "pdb,md",
    n_chains: int = 5,
    chains: str = "",
    gpu: str = "L4",
) -> None:
    """Launch one Modal call per requested checkpoint.

    Args:
        split: ATLAS split (``val`` or ``test``). Ignored when ``chains`` is set.
        checkpoints: Comma-separated subset of ``pdb,md``.
        n_chains: How many chains from the start of the split to sample. Ignored
            when ``chains`` is set.
        chains: Comma-separated explicit chain list, e.g. ``6dlm_A,6cka_B``.
            Bypasses the premval split loader; useful for running a curated
            subset (e.g. only chains under a length threshold).
        gpu: Modal GPU spec (``L4``, ``A10G``, ``A100-40GB``, ``A100-80GB``,
            ``H100``); overrides the function-level default.
    """
    if chains:
        chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    else:
        from premval.data import load_test_chains, load_val_chains

        all_chains = load_test_chains() if split == "test" else load_val_chains()
        chain_list = all_chains[:n_chains]
    print(f"split={split} chains={chain_list} gpu={gpu}")
    sample_fn = sample.with_options(gpu=gpu)
    for ck in (c.strip() for c in checkpoints.split(",") if c.strip()):
        if ck not in WEIGHTS:
            raise SystemExit(f"unknown checkpoint {ck!r}; choose from {sorted(WEIGHTS)}")
        print(f"--- launching checkpoint={ck} ---")
        sample_fn.remote(ck, split, chain_list)
    print("done. pull samples with:")
    print("    modal volume get premval-samples samples ~/.cache/premval/")
