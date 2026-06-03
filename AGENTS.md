# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. Read this
before non-trivial work; the timeless engineering rules live in
[`CODING_STANDARDS.md`](CODING_STANDARDS.md).

## Project

`premval` — a neutral, reproducible benchmark and leaderboard for **protein
conformational-ensemble generators** (AlphaFlow/ESMFlow, BioEmu, ESMDiff, …),
scored against the [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) molecular-dynamics
dataset with one fixed metric panel. Live leaderboard:
https://premval--web.modal.run/

Python 3.12+, src-layout, built with `poetry-core`. The package lives at
`src/premval/` and is declared via
`[tool.poetry] packages = [{include = "premval", from = "src"}]` in
`pyproject.toml` (the src layout is not auto-discovered by poetry-core).

## Setup & commands

A project virtualenv at `.venv/` has `premval`, dev tools, and the Modal client
installed. **Prefer `.venv/bin/<tool>` over the shell PATH** — the global tools
(e.g. the `uv tool`-installed `modal`) run under their own isolated Python and
will not see project deps.

```bash
pip install -e ".[dev]"          # editable install with dev tools
.venv/bin/pytest                 # full suite (also: pytest path::test for one)
.venv/bin/ruff check .           # lint
.venv/bin/ruff format .          # format
.venv/bin/mypy                   # strict type-check over src + tests
```

Optional extras: `viz` (matplotlib), `viz-pymol` (PyMOL renderer), `web`
(FastAPI dashboard). The core scoring path needs none of them.

`pytest`, `ruff check`, and `mypy` are the source of truth — verify with them,
not by inspection. `mypy --strict` must pass; new public APIs are annotated
(`py.typed` ships to downstream consumers).

## Architecture

The hard invariant: **the dependency arrow points one way.** `inference/`
imports `premval`, never the reverse, which keeps the installable package
CPU-only and GPU-free.

- **`src/premval/`** — the CPU-only scoring library + CLI:
  - `data/` — data access and caching. `atlas.py` (reference bundles +
    trajectories + the reference playback view), `samples.py` (model sample
    ensembles + viewer sidecars), `references.py` (reference observables),
    `published.py` (download published ensembles), `rcsb.py` (entry metadata).
  - `metrics/` — the metric panel, ported from AlphaFlow's eval scripts so
    numbers line up with the literature (`alphaflow_port.py`, `panel.py`).
  - `scoring.py` (`score_chain`), `leaderboard.py` (`score_split`,
    `load_leaderboard`), `models.py` (display metadata + held-out labels).
  - `web/` — the FastAPI dashboard (`app.py` factory `create_app`, `templates/`).
  - `cli.py` — the `premval` entry point; thin, delegates to the modules above.
- **`inference/`** — run-yourself GPU (Modal) harnesses for models PREMVAL can't
  just download. `web_modal.py` deploys the dashboard (see below).
- **`results/`** — committed per-model scores (`{model}.json`); the leaderboard
  auto-discovers any model with a results file here.
- **`data/`** — ATLAS split lists and `contamination_labels.yaml` (label
  evidence; the leaderboard renders labels from `models.py`).

### CLI / data flow

```bash
premval fetch                       # ATLAS bundles -> ~/.cache/premval/atlas/
premval ingest --model <key>        # published ensembles -> ~/.cache/premval/samples/
premval prepare-refs                # reference .npz + .view.pdb caches
premval prepare-samples             # per-(model,chain) viewer sidecars
premval score-all --split test      # batch-score into results/{model}.json
premval score --chain X --submission e.pdb   # one ad-hoc submission
premval serve --port 8000           # local dashboard (needs [web] extra)
```

`premval --help` and `premval <command> --help` document everything.

### Caches (local layout)

- ATLAS — `~/.cache/premval/atlas/`: `{kind}/{chain}.zip` bundles,
  `references/{kind}/{chain}.npz` observables, `references/{kind}/{chain}.view.pdb`
  reference playback ensembles.
- Samples — `~/.cache/premval/samples/`: `{model}/{chain}.pdb` raw ensembles and
  the `_view/` / `_observables/` / `_metrics/` viewer sidecars (`_aligned/` is an
  intermediate). Underscore-prefixed dirs are not treated as models.

## Held-out (contamination) labels

Every model carries a label describing how strong its ATLAS held-out guarantee
is, defined in `src/premval/models.py` (`Contamination` literal +
`MODEL_REGISTRY`), rendered as a colored badge on the leaderboard:

- `held_out` (green) — never trained on ATLAS/MD (e.g. BioEmu, PDB-only variants).
- `weak_holdout` (amber) — fine-tuned on the ATLAS train split, scored on the
  held-out test split, but the split is temporal only (AlphaFlow-MD, ESMFlow-MD).
- `uncertain` (grey) — training relationship to ATLAS not established (ESMDiff).

`data/contamination_labels.yaml` records the evidence (and labels for run-yourself
models not yet scored); keep it consistent with `models.py`.

## Web dashboard on Modal

`inference/web_modal.py` serves `premval.web.app:create_app` as a Modal ASGI app
(`modal deploy inference/web_modal.py`, after `set -a; source .env; set +a`).
Deployed in the **`premval`** Modal workspace at **https://premval--web.modal.run**.
It is CPU-only — no model sampling is reachable from the public URL.

Two read-only Modal Volumes back it: `premval-cache` (`/cache`, the ATLAS cache)
and `premval-samples` (`/samples`, the viewer sidecars). The served app does not
persist its own volume writes, so everything it shows must be **precomputed and
uploaded** (`prepare-refs` + `prepare-samples`, then `modal volume put`). Two
load-bearing gotchas:

- **Never add `GZipMiddleware`** to the FastAPI app: behind Modal's edge proxy it
  truncates large PDB responses mid-stream.
- The raw `samples/{model}/{chain}.pdb` ensembles are **not** uploaded;
  `available_models`/`available_chains` discover models from the `_view` sidecars,
  and `/ensemble.pdb` serves the precomputed `.view.pdb` (rebuilding from the
  bundle per request costs 18-42 s).

## Conventions

- Ruff: line length 100, target `py312`, lint rules `E, F, I, W, B, UP`.
- Google-style docstrings, scaled to the function; comments explain *why*, not
  *what*. Keep public APIs annotated (`py.typed`).
- Tests live alongside changes; a function without a test has an undefined
  contract.

See [`CODING_STANDARDS.md`](CODING_STANDARDS.md) for the full rulebook (DRY,
YAGNI, KISS, SOLID, and the agent-specific guardrails).
