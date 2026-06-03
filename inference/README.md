# inference/ — run-yourself GPU harnesses

This directory holds the GPU harness scripts that generate protein ensembles
for models PREMVAL cannot just download. It populates the leaderboard with
open-weight models on the ATLAS val/test splits.

## Hard constraint: `premval` stays CPU-only

The importable `premval` package is CPU-only and free of GPU dependencies.
Everything in `inference/` lives **outside** the package:

- `premval` never imports `inference/` (the dependency only goes one way:
  harnesses import `premval` for `sample_path`, never the reverse).
- Each harness pins its own model stack (clone/conda/pip) and imports the
  heavy GPU deps **lazily**, inside the functions that need them, so
  `--self-test` and `--help` run on a plain CPU `premval` install.
- No harness dependency goes into `pyproject.toml`. The model stacks are
  documented here and installed by hand into a separate environment.
- The only thing a harness writes is multi-model PDBs into PREMVAL's samples
  cache. It produces no other artifacts the package depends on.

## Shared script contract

All three harnesses (`str2str_run.py`, `bioemu_run.py`, `confdiff_run.py`)
follow one CLI:

```bash
python inference/<model>_run.py --split {val,test} \
    [--chains CHAIN ...] \
    [--out-model KEY] \
    [--n-samples 250] \
    [--samples-dir DIR] \
    [--self-test]
```

- `--split` selects the ATLAS split (val = 39 chains, test = 82 chains).
- `--chains` overrides the split with an explicit chain list.
- `--out-model` sets the samples-cache key (the leaderboard / results
  filename). Defaults are per model below.
- `--n-samples` is the number of frames per chain (default 250).
- `--samples-dir` overrides the cache root; otherwise the harness reads
  `PREMVAL_SAMPLES_DIR`, then falls back to `~/.cache/premval/samples/`.

**Output.** Each chain is written as a single 250-MODEL multi-model PDB to
`sample_path(out_model, chain)` →
`~/.cache/premval/samples/{out_model}/{chain}.pdb`. This is exactly where
`premval score-all` looks, so no extra ingest step is needed for harness
output.

**`--self-test` (no GPU).** This runs the full CPU input/output path with
synthetic frames: it loads the chain topology/sequence, builds a trajectory of
exactly `--n-samples` random frames, writes the multi-model PDB to the cache,
and reloads it to assert the frame count round-trips. It exercises everything
except the GPU sampler, so it is how you validate a harness on a laptop before
booking GPU time. Point it at a scratch dir to avoid polluting the cache:

```bash
PREMVAL_SAMPLES_DIR=$(mktemp -d) python inference/str2str_run.py --self-test
```

## Run telemetry

Every harness records basic telemetry per chain via the shared
[`common.py`](common.py) helper and writes a JSON sidecar next to each sample
PDB: `~/.cache/premval/samples/{out_model}/{chain}.telemetry.json`. It captures
wall time (and per-sample seconds) and, when a CUDA device is present, peak and
time-averaged GPU memory — the same numbers a wandb/Modal system panel would
surface, captured locally so the harnesses keep their no-extra-deps,
no-account, no-network property. Fields:

| field | meaning |
|-------|---------|
| `chain`, `n_samples` | chain id and frames generated |
| `wall_seconds`, `seconds_per_sample` | sampling wall time and its per-frame rate |
| `device` | CUDA device name, or `cpu` when no GPU was used |
| `gpu_peak_mb`, `gpu_mean_mb` | peak / background-polled mean allocated GPU memory (`null` on CPU) |
| `gpu_poll_count` | number of memory polls (0 on CPU, or if sampling finished within one poll interval) |

A one-line summary is also printed to stdout per chain. The sidecars are the
machine-readable source of truth: a later dashboard reader can aggregate them
without re-running anything. `--self-test` writes a sidecar too (CPU, so the
GPU fields are `null`), which is how the telemetry path is verified without a
GPU.

### Optional: Weights & Biases

The same per-chain telemetry can also stream to a wandb run. It is **off by
default** and enabled purely by environment — set `WANDB_PROJECT` (and
authenticate with `WANDB_API_KEY` or a prior `wandb login`) before launching a
harness:

```bash
export WANDB_API_KEY=...        # or run `wandb login` once
export WANDB_PROJECT=premval-inference
python inference/bioemu_run.py --split val
```

Each harness invocation becomes one wandb run named `{out_model}-{split}`;
every chain logs `wall_seconds`, `seconds_per_sample`, and (on GPU)
`gpu_peak_mb` / `gpu_mean_mb`, and wandb additionally captures its own
GPU/system-metrics panel for the duration. When `WANDB_PROJECT` is unset the
logger is a no-op, the JSON sidecars are unaffected, and `wandb` need not be
installed (it is imported lazily, only when enabled).

## Per-model harnesses

### Str2Str (`str2str_run.py`)

- **Upstream:** https://github.com/lujiarui/Str2Str (MIT).
- **Weights:** Google Drive checkpoint linked from the repo README.
- **Method:** zero-shot, score-based sampler. Perturbs/anneals the chain's
  ATLAS topology PDB; no sequence model, no MD training.
- **Env setup:**
  ```bash
  git clone https://github.com/lujiarui/Str2Str /path/to/Str2Str
  cd /path/to/Str2Str && conda env create -f env.yaml   # see repo README
  # download the GDrive checkpoint into the repo's checkpoint dir per README
  ```
- **Run:**
  ```bash
  python inference/str2str_run.py --split val --str2str-repo /path/to/Str2Str
  python inference/str2str_run.py --split test --str2str-repo /path/to/Str2Str
  ```
  (`--str2str-repo` may instead be supplied via the `STR2STR_REPO` env var.)
- **GPU + cost:** the cheapest of the three; a single small/mid GPU
  (e.g. one consumer card or a slice of an A100) clears both splits quickly,
  20–100x faster than diffusion baselines per the paper.
- **Contamination label:** `test` (zero-shot PDB sampler; no MD/ATLAS in
  training). `out-model = str2str`.

### BioEmu (`bioemu_run.py`)

- **Upstream:** https://github.com/microsoft/bioemu (MIT) and
  https://github.com/microsoft/bioemu-benchmarks.
- **Weights:** https://huggingface.co/microsoft/bioemu (auto-downloaded by the
  `bioemu` package on first use).
- **Method:** sequence-conditioned emulator; samples structures from the
  chain's sequence. See the `md_emulation` benchmark in
  `microsoft/bioemu-benchmarks` (50 ATLAS cases; ATLAS is *not* in BioEmu's
  training set).
- **Env setup:**
  ```bash
  pip install bioemu          # Linux, Python 3.10+; pulls the [cuda] stack
  # HF weights auto-download on first sample call
  ```
- **Run:**
  ```bash
  python inference/bioemu_run.py --split val
  python inference/bioemu_run.py --split test
  ```
- **GPU + cost:** fast on a single GPU (A100-80GB class recommended in the
  upstream README); both splits finish on one card.
- **Contamination label:** `uncertain` (broad MD+AFDB+stability corpus; ATLAS
  overlap unclear, though ATLAS is reported not to be in training).
  `out-model = bioemu`.

### ConfDiff (`confdiff_run.py`)

- **Upstream:** https://github.com/bytedance/ConfDiff (Apache-2.0).
- **Weights:** https://huggingface.co/leowang17/ConfDiff (includes both the
  PDB base checkpoint and the ATLAS-fine-tuned `-MD` checkpoint).
- **Method:** sequence-conditioned, ESMFold-based diffusion. Runs on a single
  GPU.
- **Env setup:**
  ```bash
  git clone https://github.com/bytedance/ConfDiff /path/to/ConfDiff
  cd /path/to/ConfDiff && conda env create -f environment.yml   # see repo README
  # download the leowang17/ConfDiff checkpoints from HF (base and -MD)
  ```
- **Run (two checkpoints, two cache keys):**
  ```bash
  # PDB base checkpoint -> out-model confdiff
  python inference/confdiff_run.py --split val  --checkpoint base
  python inference/confdiff_run.py --split test --checkpoint base

  # ATLAS-fine-tuned -MD checkpoint -> out-model confdiff_md
  python inference/confdiff_run.py --split val  --checkpoint md
  python inference/confdiff_run.py --split test --checkpoint md
  ```
  `--checkpoint base` writes `out-model = confdiff`; `--checkpoint md` writes
  `out-model = confdiff_md`.
- **GPU + cost:** single GPU; comparable to the other ESMFold-based samplers.
- **Contamination labels:**
  - `--checkpoint base` → `confdiff`, label `test` (ESMFold/PDB-trained; ATLAS
    MD not in training).
  - `--checkpoint md` → `confdiff_md`, label `in_distribution` (the `-MD`
    checkpoint is fine-tuned on ATLAS).

### AlphaFlow distilled (`alphaflow_run.py`)

- **Upstream:** https://github.com/bjing2016/alphaflow (MIT).
- **Weights:** https://huggingface.co/bjing-mit/alphaflow (this harness
  defaults to the *distilled* checkpoints; the base diffusion variants use the
  same `predict.py` and can be wired in by editing `_alphaflow_command`).
- **Method:** AF2 repurposed as a flow-matching ensemble sampler; needs MSAs
  (`{chain}.a3m`). The distilled checkpoints sample in one step (paper
  reports ~10x cheaper, comparable quality).
- **Env setup:**
  ```bash
  git clone https://github.com/bjing2016/alphaflow /path/to/alphaflow
  cd /path/to/alphaflow && conda env create -f environment.yml   # see repo README
  huggingface-cli download bjing-mit/alphaflow \
      alphaflow_pdb_distilled_202402.pt alphaflow_md_distilled_202402.pt \
      --local-dir ./weights
  export ALPHAFLOW_REPO=/path/to/alphaflow
  export ALPHAFLOW_PDB_DISTILLED_CKPT=/path/to/alphaflow/weights/alphaflow_pdb_distilled_202402.pt
  export ALPHAFLOW_MD_DISTILLED_CKPT=/path/to/alphaflow/weights/alphaflow_md_distilled_202402.pt
  export ALPHAFLOW_MSA_DIR=/path/to/atlas_msa_dir       # {chain}.a3m per chain
  ```
- **Run (two checkpoints, two cache keys):**
  ```bash
  # PDB-trained distilled -> out-model alphaflow_pdb_distilled
  python inference/alphaflow_run.py --split val  --checkpoint pdb
  python inference/alphaflow_run.py --split test --checkpoint pdb

  # ATLAS-fine-tuned distilled -> out-model alphaflow_md_distilled
  python inference/alphaflow_run.py --split val  --checkpoint md
  python inference/alphaflow_run.py --split test --checkpoint md
  ```
- **GPU + cost:** single A100-80GB recommended; distilled is roughly an order
  of magnitude cheaper than the base diffusion path. See the cost-estimate
  table for ballparks.
- **Contamination labels:**
  - `--checkpoint pdb` → `alphaflow_pdb_distilled`, label `test` (PDB-trained
    only).
  - `--checkpoint md` → `alphaflow_md_distilled`, label `in_distribution`
    (ATLAS-fine-tuned).

### ESMFlow distilled (`esmflow_run.py`)

- **Upstream:** same repo as AlphaFlow (https://github.com/bjing2016/alphaflow,
  MIT). The same `predict.py` drives both modes; only `--mode esmfold` and the
  weights file differ.
- **Weights:** https://huggingface.co/bjing-mit/alphaflow (distilled ESMFlow
  variants).
- **Method:** ESMFold repurposed as a flow-matching ensemble sampler.
  Single-sequence (no MSAs). The distilled checkpoints sample in one step.
- **Env setup:** same conda env and clone as the AlphaFlow harness, plus the
  ESMFlow weights and env vars (no MSA dir needed):
  ```bash
  huggingface-cli download bjing-mit/alphaflow \
      esmflow_pdb_distilled_202402.pt esmflow_md_distilled_202402.pt \
      --local-dir ./weights
  export ALPHAFLOW_REPO=/path/to/alphaflow                  # shared with alphaflow_run.py
  export ESMFLOW_PDB_DISTILLED_CKPT=/path/to/alphaflow/weights/esmflow_pdb_distilled_202402.pt
  export ESMFLOW_MD_DISTILLED_CKPT=/path/to/alphaflow/weights/esmflow_md_distilled_202402.pt
  ```
- **Run (two checkpoints, two cache keys):**
  ```bash
  # PDB-trained distilled -> out-model esmflow_pdb_distilled
  python inference/esmflow_run.py --split val  --checkpoint pdb
  python inference/esmflow_run.py --split test --checkpoint pdb

  # ATLAS-fine-tuned distilled -> out-model esmflow_md_distilled
  python inference/esmflow_run.py --split val  --checkpoint md
  python inference/esmflow_run.py --split test --checkpoint md
  ```
- **GPU + cost:** cheaper than AlphaFlow (no MSA path; smaller backbone). A
  single A100-80GB clears both splits comfortably.
- **Contamination labels:**
  - `--checkpoint pdb` → `esmflow_pdb_distilled`, label `test`.
  - `--checkpoint md` → `esmflow_md_distilled`, label `in_distribution`.

## val-before-test rule

Always smoke the **39 val chains first**, eyeball that the metrics are sane
(`premval score-all --split val`), and only then commit GPU time to the
**82 test chains**. Re-running is **resumable**: a harness skips any chain that
already has a PDB in the samples cache, so an interrupted test run picks up
where it left off (delete the chain's PDB to force a re-sample).

## No-GPU runbook: download-available published models

These models publish their ATLAS ensembles, so PREMVAL ingests their
coordinates and re-runs its own metric panel on them. This is the fair, free
path (no GPU needed): the same metrics are computed for downloaded and
self-run models alike.

```bash
# EBA: 250-sample multi-model CIF target. Confirm EBA's target names actually
# map to ATLAS chain ids and log any mismatches before trusting the scores.
premval ingest --model eba

# AlphaFlow / ESMFlow PDB bases (PDB-trained, no ATLAS MD).
premval ingest --model alphaflow_pdb_base
premval ingest --model esmflow_pdb_base

# alphaflow_md_base and esmflow_md_base are already ingested + scored.

# Score the ingested coordinates.
premval score-all --split val
premval score-all --split test
```

## Standard post-run step (any harness or ingest)

After a harness fills the cache (or after an ingest), score it into
`results/<key>.json`:

```bash
premval score-all --models <key> --split val
premval score-all --models <key> --split test
```

## Contamination labels

Per-model ATLAS contamination labels live in
[`../data/contamination_labels.yaml`](../data/contamination_labels.yaml), with a
one-line `basis` field recording the evidence behind each call. The keys there
match the samples-cache directory / `results/` filename so the leaderboard joins
labels to scores directly.
