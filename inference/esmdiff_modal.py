"""Modal wrapper around ``inference/esmdiff_run.py``.

Builds a CUDA image that clones the ESMDiff repo
(https://github.com/lujiarui/esmdiff), installs its pinned stack (Python 3.10,
torch, ``esm==3.0.4``, deepspeed, hydra/lightning), bakes in the ESMDiff
checkpoint (``release_v0.pt`` from the README's Google Drive), mounts the local
PREMVAL checkout, and runs ``esmdiff_run.py`` against the ATLAS val/test splits.
Generated samples land on the shared ``premval-samples`` Modal Volume under
``samples/esmdiff/{chain}.pdb`` plus the matching ``{chain}.telemetry.json``
sidecar, mirroring the local cache layout so ``modal volume get`` syncs them
straight into ``~/.cache/premval/samples/`` (same Volume the BioEmu run used).

Gated ESM3 weights
------------------
ESMDiff samples by decoding ESM3 structure tokens, so it downloads the **gated**
``EvolutionaryScale/esm3`` weights from HuggingFace at runtime. That needs your
HF token. ``modal.Secret.from_dotenv`` reads the repo-root ``.env`` (which holds
``HF_TOKEN`` alongside the Modal tokens) and injects it into the container, so no
separate ``modal secret create`` step is needed; the token never lands in the
image. The token's account must have accepted the ESM3 non-commercial license at
https://huggingface.co/EvolutionaryScale/esm3 (you have), and ``huggingface_hub``
picks ``HF_TOKEN`` up for the gated pull.

``premval`` itself is never imported on Modal: the script is invoked as a
subprocess (same contract as a local GPU run), so this file and ``esmdiff_run``
are the only GPU-aware code paths on the wrapper side.

Launch
------
::

    set -a; source .env; set +a   # MODAL_TOKEN_ID/SECRET (HF_TOKEN read from .env)
    modal run inference/esmdiff_modal.py                       # 1 chain, smoke test
    modal run inference/esmdiff_modal.py --split val           # whole val split
    modal run inference/esmdiff_modal.py --split test --gpu A100-40GB

After the run, pull the samples back locally::

    modal volume get premval-samples samples ~/.cache/premval/

The Volume only ever grows; re-launches skip chains whose PDB already exists.

If the Google-Drive checkpoint fetch ever fails at build time (Drive quota), put
the file on the Volume instead and point ``--ckpt`` at it::

    modal volume put premval-samples release_v0.pt ckpt/release_v0.pt
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

ESMDIFF_REPO = "/opt/esmdiff"
# README's Google Drive checkpoint (release_v0.pt), placed where the repo's
# config expects it (data/ckpt/).
_CKPT_DRIVE_ID = "1p99hxxfgIlLlO1i0CP-P34Rjnegkcb_L"
_CKPT_PATH = f"{ESMDIFF_REPO}/data/ckpt/release_v0.pt"

# torch goes in first so deepspeed (in the repo's requirements.txt) compiles
# against an already-present torch instead of pulling its own. scikit-learn and
# requests are lazy deps of premval.data (PCA in references.py; the published.py
# downloader); the rest (esm==3.0.4, mdtraj, lightning, hydra) come from the
# repo's pinned requirements.txt. gdown is build-only, to fetch the checkpoint.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install("torch>=2.0.0", "torchvision>=0.15.0")
    .run_commands(
        f"git clone --depth 1 https://github.com/lujiarui/esmdiff.git {ESMDIFF_REPO}",
        f"pip install -r {ESMDIFF_REPO}/requirements.txt",
        f"pip install -e {ESMDIFF_REPO}",
    )
    .pip_install("gdown", "scikit-learn", "requests")
    .run_commands(
        f"mkdir -p {ESMDIFF_REPO}/data/ckpt",
        f'gdown "https://drive.google.com/uc?id={_CKPT_DRIVE_ID}" -O {_CKPT_PATH}',
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

app = modal.App("premval-esmdiff")
samples_volume = modal.Volume.from_name("premval-samples", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # default; ESM3-open is small. Per-invocation override via with_options.
    volumes={"/cache": samples_volume},
    secrets=[modal.Secret.from_dotenv(REPO_ROOT)],
    timeout=60 * 60 * 6,
)
def sample(
    split: str,
    chains: list[str],
    n_samples: int,
    mode: str = "ddpm",
    num_steps: int = 25,
) -> None:
    """Run ``esmdiff_run.py`` over a list of chains.

    The Modal Volume is committed at exit so partial progress survives a crash,
    and the harness's skip-if-PDB-exists rule keeps re-runs idempotent. HF_HOME
    is container-local (not on the shared Volume): the gated ESM3 weights
    re-download per container, but parallel shards never race on the same cache
    files. Each shard's samples land under distinct per-chain paths, so the
    Volume itself is written without conflict. The ``huggingface`` secret
    supplies ``HF_TOKEN`` for the gated ESM3 pull.
    """
    import os
    import subprocess

    env = dict(os.environ)
    env.update(
        {
            "PREMVAL_SAMPLES_DIR": "/cache/samples",
            "ESMDIFF_REPO": ESMDIFF_REPO,
            "HF_HOME": "/root/hf",
            "PYTHONPATH": "/opt/premval/src",
            # ESMDiff auto-chunks samples by a hardcoded residue-square budget
            # tuned for an A100; on smaller cards the resident ESM3 weights plus
            # one chunk fragment the allocator into an OOM that's only a few
            # hundred MB over. expandable_segments reclaims the reserved-but-
            # unallocated slack the OOM message flags (see esmflow_modal.py).
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    cmd = [
        "python",
        "/opt/premval/inference/esmdiff_run.py",
        "--split",
        split,
        "--n-samples",
        str(n_samples),
        "--mode",
        mode,
        "--num-steps",
        str(num_steps),
        "--esmdiff-repo",
        ESMDIFF_REPO,
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
    n_chains: int = 1,
    chains: str = "",
    gpu: str = "A10G",
    n_samples: int = 250,
    mode: str = "ddpm",
    num_steps: int = 25,
    timeout_hours: float = 6.0,
) -> None:
    """Launch one Modal call sampling the requested chains with ESMDiff.

    Args:
        split: ATLAS split (``val`` or ``test``). Ignored when ``chains`` is set.
        n_chains: How many chains from the start of the split to sample. Ignored
            when ``chains`` is set. Defaults to 1 (a cheap smoke test).
        chains: Comma-separated explicit chain list, e.g. ``6e33_A,6dlm_A``.
            Bypasses the premval split loader.
        gpu: Modal GPU spec (``L4``, ``A10G``, ``A100-40GB``, ``A100-80GB``,
            ``H100``); overrides the function-level default.
        n_samples: Frames per chain (the leaderboard contract is 250).
        mode: ESMDiff sampler, ``ddpm`` (the paper's flagship) or ``gibbs``.
        num_steps: Diffusion steps per sample.
        timeout_hours: Per-call Modal timeout.
    """
    if chains:
        chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    else:
        from premval.data import load_test_chains, load_val_chains

        all_chains = load_test_chains() if split == "test" else load_val_chains()
        chain_list = all_chains[:n_chains]
    print(
        f"split={split} chains={chain_list} gpu={gpu} n_samples={n_samples} "
        f"mode={mode} num_steps={num_steps} timeout_hours={timeout_hours}"
    )
    sample.with_options(gpu=gpu, timeout=int(timeout_hours * 3600)).remote(
        split, chain_list, n_samples, mode, num_steps
    )
    print("done. pull samples with:")
    print("    modal volume get premval-samples samples ~/.cache/premval/")
