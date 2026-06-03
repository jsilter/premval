# PREMVAL — Protein Ensemble Evaluation

A neutral, reproducible benchmark and leaderboard for **protein conformational
ensemble generators** (AlphaFlow/ESMFlow, BioEmu, ESMDiff, ConfDiff, Str2Str,
…), scored on the [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) molecular-dynamics
dataset with a single fixed metric panel.

**Live leaderboard:** https://premval--web.modal.run/

The field moves fast and every paper benchmarks its own model, on its own
systems, with its own metrics; there is no neutral cross-model comparison.
PREMVAL is that comparison: one metric panel, one submission format, one
reference dataset, applied identically to every model.

Its differentiator is **per-model held-out labels**. AlphaFlow-MD and ESMFlow-MD
fine-tune on the ATLAS train split and are scored on the held-out test split, but
that split is temporal only (no sequence-homology filter), so the guarantee is
weak; BioEmu never trains on ATLAS at all; ESMDiff's training relationship to
ATLAS is unclear. Models trained on ATLAS can score better partly because the
test split resembles their training data, and no individual paper surfaces this.
PREMVAL tags every model `held_out` / `weak_holdout` / `uncertain` so the
leaderboard is read with that context, not without it.

The scoring library is **CPU-only and free of GPU dependencies**: it ingests
already-generated ensembles (downloaded or produced by the GPU harnesses in
[`inference/`](inference/)) and re-runs the same metrics on all of them.

## Models & contamination labels

Every evaluated model is tagged by how strong its held-out guarantee is on the
ATLAS evaluation, because a model whose test data resembles its training set
holds an advantage that raw scores hide:

- **`held_out`** (green) — never trained on ATLAS or MD data (PDB-only,
  zero-shot, or training filtered away from the test proteins). A genuine
  held-out evaluation.
- **`weak_holdout`** (amber) — fine-tuned on the ATLAS *train* split and scored
  on the held-out *test* split, but the split is temporal only (no
  sequence-homology filter), so test homologs may resemble training data. A real
  but weak held-out guarantee; read these scores with that caveat.
- **`uncertain`** (grey) — the training corpus's relationship to ATLAS is not
  established (a broad MD/structure corpus, or a fine-tune of a model whose
  pretraining overlap is unclear).

Models currently on the leaderboard:

| Model | What it is | Label |
|-------|------------|-------|
| [AlphaFlow-MD](https://arxiv.org/abs/2402.04845) (base, distilled) | AlphaFold2 fine-tuned with flow matching on ATLAS MD (Jing et al., ICML 2024) | `weak_holdout` |
| [ESMFlow-MD](https://arxiv.org/abs/2402.04845) (base, distilled) | ESMFold fine-tuned with flow matching on ATLAS MD (Jing et al., ICML 2024) | `weak_holdout` |
| [BioEmu](https://doi.org/10.1101/2024.12.05.626885) | Equilibrium-ensemble emulator; not trained on ATLAS, training filtered to <40% sequence identity to test proteins (Lewis et al., 2024) | `held_out` |
| [ESMDiff](https://arxiv.org/abs/2410.18403) | ESM3 fine-tuned with masked diffusion over discrete structure tokens (Lu et al., ICLR 2025) | `uncertain` |

The leaderboard's badge labels are defined in
[`src/premval/models.py`](src/premval/models.py); per-model evidence and labels
for additional run-yourself models (Str2Str, ConfDiff, the PDB-trained flow
variants, which are `held_out`) are recorded in
[`data/contamination_labels.yaml`](data/contamination_labels.yaml) and wired up
in [`inference/`](inference/).

## The metric panel

Ported from AlphaFlow's evaluation scripts (Jing et al., ICML 2024) so numbers
line up with the literature, plus the raw RMWD components. Every metric compares
a 250-frame submission ensemble against the ATLAS MD reference for the same
chain:

| Metric (JSON key)             | What it measures                                                      |
|-------------------------------|----------------------------------------------------------------------|
| `rmsf_pearson`                | Per-residue Cα flexibility (RMSF) correlation with MD (higher better) |
| `rmwd`                        | Root-mean Wasserstein distance between per-atom Gaussian fits (lower) |
| `emd_mean_rms` / `emd_var_rms`| RMWD split into mean-displacement and covariance-mismatch components  |
| `md_pca_w2`                   | 2-Wasserstein distance in the MD-fit PCA basis (lower better)         |
| `weak_contacts_jaccard`       | Jaccard overlap of weak (transiently-broken) Cα–Cα contacts (higher) |
| `transient_contacts_jaccard`  | Jaccard overlap of transiently-formed Cα–Cα contacts (higher)        |

The leaderboard summarizes each quality metric by its **mean** across a split,
and per-chain inference wall time by its **median** (runtime is dominated by
sequence length, so it is heavily right-skewed).

## Submission format

One ensemble per chain, as a single **multi-model PDB with exactly 250 frames**,
named `{chain}.pdb` (e.g. `6cka_B.pdb`). The reference splits are ATLAS:
**39 val chains** and **82 test chains**.

## Install

Python 3.12+. Editable install with dev tools:

```bash
pip install -e ".[dev]"
```

Optional extras: `viz` (matplotlib), `viz-pymol` (PyMOL renderer), `web`
(FastAPI dashboard). The core scoring path needs none of them.

## Quickstart

```bash
# 1. Download ATLAS reference bundles into the local cache (~/.cache/premval).
premval fetch                      # val split; --chains ... for specific chains

# 2a. Ingest a model that publishes its ATLAS ensembles (no GPU).
premval ingest --model alphaflow_pdb_base
# 2b. ...or generate one yourself on a GPU; see inference/README.md.

# 3. Precompute the reference-observables cache, then batch-score a split.
premval prepare-refs
premval score-all --split val      # writes results/{model}.json

# 4. Score a single submission ad hoc.
premval score --chain 6cka_B --submission ensemble.pdb

# 5. Serve the dashboard locally (requires the [web] extra).
premval serve --port 8000
```

`premval --help` (and `premval <command> --help`) documents every subcommand.

## Repository layout

| Path                  | What it holds                                                        |
|-----------------------|---------------------------------------------------------------------|
| `src/premval/`        | The CPU-only scoring library + CLI (`metrics/`, `data/`, `scoring.py`, `leaderboard.py`, `web/`) |
| `inference/`          | Run-yourself GPU harnesses for models PREMVAL can't just download (see its [README](inference/README.md)) |
| `results/`            | Committed per-model scores (`{model}.json`); the leaderboard reads these |
| `data/`               | ATLAS split lists and `contamination_labels.yaml`                   |
| `inference/web_modal.py` | Deploys the dashboard as a Modal ASGI app (the live `premval--web.modal.run`) |
| `tests/`              | Pytest suite (run before every change)                              |

The dependency arrow only points one way: `inference/` imports `premval`, never
the reverse, which keeps the installable package GPU-free.

## Development

```bash
pytest               # full suite
ruff check . && ruff format .
mypy                 # strict mode over src + tests
```

`mypy --strict` and the lint rules (`E, F, I, W, B, UP`, line length 100) must
pass. See [`CODING_STANDARDS.md`](CODING_STANDARDS.md) for the conventions.

## Contributing a model

Score your ensemble into `results/{model}.json` (via `premval ingest` +
`premval score-all`, or a GPU harness in `inference/`), add a row to
[`data/contamination_labels.yaml`](data/contamination_labels.yaml) and a display
entry to `src/premval/models.py`, and open a PR. The leaderboard auto-discovers
any model with a committed results file.

## License & attribution

PREMVAL is released under the **MIT License**. The metric panel is ported from
[AlphaFlow](https://github.com/bjing2016/alphaflow) (Jing et al., "AlphaFold
Meets Flow Matching for Generating Protein Ensembles," ICML 2024). Each model
evaluated here carries its own upstream weights license (documented in
[`inference/README.md`](inference/README.md)); ESMDiff in particular depends on
the non-commercial, gated ESM3 weights. ATLAS reference data is distributed by
its authors under their own terms.
