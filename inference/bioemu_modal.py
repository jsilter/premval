"""Modal wrapper around ``inference/bioemu_run.py``.

Builds a CUDA-capable image with the ``bioemu`` package (and the handful of
deps the harness's ``premval.data`` import needs), mounts the local PREMVAL
checkout, and runs ``bioemu_run.py`` against the ATLAS val/test splits.
Generated samples land on the shared ``premval-samples`` Modal Volume under
``samples/bioemu/{chain}.pdb`` plus the matching ``{chain}.telemetry.json``
sidecar, mirroring the local cache layout so ``modal volume get`` syncs them
straight into ``~/.cache/premval/samples/`` (same Volume the ESMFlow run used).

Unlike ``esmflow_modal.py`` (which fights the alphaflow CUDA-11.8/torch-1.12
stack), BioEmu installs cleanly from PyPI on a modern torch; its weights and
the ESM2 embeddings it conditions on download from HuggingFace on first use.
``HF_HOME`` is pointed at the Volume so that download happens once and is
reused across chains and re-launches.

``premval`` itself is never imported on Modal: the script is invoked as a
subprocess (same contract as a local GPU run), so this file is the only
GPU-aware code path on the wrapper side.

Launch
------
::

    set -a; source .env; set +a   # MODAL_TOKEN_ID/SECRET
    modal run inference/bioemu_modal.py                       # 1 chain, smoke test
    modal run inference/bioemu_modal.py --split val           # whole val split
    modal run inference/bioemu_modal.py --split test --gpu A100-80GB

After the run, pull the samples back locally::

    modal volume get premval-samples samples ~/.cache/premval/

The Volume only ever grows; re-launches skip chains whose PDB already exists.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

# BioEmu installs a modern torch with bundled CUDA wheels, so a plain slim
# image on a GPU worker is enough. scikit-learn is a lazy dependency of
# premval.data (PCA in references.py); mdtraj is needed by the harness to load
# BioEmu's xtc/pdb pair and re-emit a multi-model PDB. bioemu pulls in numpy.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # bioemu[cuda] pulls jax[cuda12]==0.4.35 so the AlphaFold/ColabFold sequence
    # embedding (a jax/Evoformer model) runs on the GPU; plain `bioemu` installs
    # a CPU-only jaxlib and the embedding crawls on long sequences.
    #
    # bioemu hard-pins jax==0.4.35 but leaves tensorflow-cpu unbounded
    # (>=2.12.0). Unpinned, pip grabs the latest TF (2.21.0), whose XLA/protobuf
    # descriptors collide with jaxlib 0.4.35's the moment both load -> SIGABRT
    # ("File already exists in database: xla/xla_data.proto"). Both are needed
    # (TF for the vendored-AlphaFold ColabFold embedding featurizer, jax for the
    # AF2 model), so pin TF to 2.18.0 (released alongside jax 0.4.35, protobuf
    # 5.x, ml-dtypes 0.4.x) so the two coexist. Resolved in one pass with the
    # harness's extra deps so protobuf doesn't get bumped back up.
    .pip_install("bioemu[cuda]", "tensorflow-cpu==2.18.0", "mdtraj", "scikit-learn", "wandb")
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

app = modal.App("premval-bioemu")
samples_volume = modal.Volume.from_name("premval-samples", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # default; per-invocation override via `sample.with_options(gpu=...)`
    volumes={"/cache": samples_volume},
    timeout=60 * 60 * 6,
)
def sample(split: str, chains: list[str], n_samples: int, max_oversample: float = 3.0) -> None:
    """Run ``bioemu_run.py`` over a list of chains.

    The Modal Volume is committed at exit so partial progress survives a crash,
    and the harness's skip-if-PDB-exists rule keeps re-runs idempotent. HF_HOME
    is container-local (not on the shared Volume): weights re-download per
    container, but parallel shards never race on the same cache files. Each
    shard's samples land under distinct per-chain paths, so the Volume itself is
    written without conflict.
    """
    import os
    import subprocess

    env = dict(os.environ)
    env.update(
        {
            "PREMVAL_SAMPLES_DIR": "/cache/samples",
            "HF_HOME": "/root/hf",
            "PYTHONPATH": "/opt/premval/src",
            # BioEmu runs the AF2 embedding (jax) and the diffusion model (torch)
            # on the same GPU. jax/XLA preallocates ~75% of VRAM by default,
            # starving torch and forcing CUDA OOM even at small batches. Make jax
            # allocate on demand so torch gets the headroom the larger batches
            # need; expandable_segments curbs torch-side fragmentation.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    cmd = [
        "python",
        "/opt/premval/inference/bioemu_run.py",
        "--split",
        split,
        "--n-samples",
        str(n_samples),
        "--max-oversample",
        str(max_oversample),
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
    max_oversample: float = 3.0,
    timeout_hours: float = 6.0,
) -> None:
    """Launch one Modal call sampling the requested chains with BioEmu.

    Args:
        split: ATLAS split (``val`` or ``test``). Ignored when ``chains`` is set.
        n_chains: How many chains from the start of the split to sample. Ignored
            when ``chains`` is set. Defaults to 1 (a cheap smoke test).
        chains: Comma-separated explicit chain list, e.g. ``6e33_A,6dlm_A``.
            Bypasses the premval split loader.
        gpu: Modal GPU spec (``L4``, ``A10G``, ``A100-40GB``, ``A100-80GB``,
            ``H100``); overrides the function-level default.
        n_samples: Frames per chain. Lower it (e.g. 20) for quick calibration
            runs that only need to reach peak GPU memory, not a full ensemble.
        max_oversample: Cap on raw samples as a multiple of ``n_samples`` (see
            ``bioemu_run._sample_chain``). Raise it (e.g. 50) to rescue chains
            whose physical-frame survival is too low for the default 3x cap.
        timeout_hours: Per-call Modal timeout. Bump it for high ``max_oversample``
            rescues that generate many thousands of samples on one chain.
    """
    if chains:
        chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    else:
        from premval.data import load_test_chains, load_val_chains

        all_chains = load_test_chains() if split == "test" else load_val_chains()
        chain_list = all_chains[:n_chains]
    print(
        f"split={split} chains={chain_list} gpu={gpu} n_samples={n_samples} "
        f"max_oversample={max_oversample} timeout_hours={timeout_hours}"
    )
    sample.with_options(gpu=gpu, timeout=int(timeout_hours * 3600)).remote(
        split, chain_list, n_samples, max_oversample
    )
    print("done. pull samples with:")
    print("    modal volume get premval-samples samples ~/.cache/premval/")
