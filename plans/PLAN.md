# PREMVAL: A Neutral Benchmark + Leaderboard for Protein Conformational Ensemble Generators

**Status**: Execution plan / actively building
**Date**: 2026-05-15
**Name**: PREMVAL — *PRotein ensEMble eVALuation*

---

## Context

The protein conformational ensemble generator field is in a stampede — BioEmu, AlphaFlow/ESMFlow, ANewSampling, JAMUN, Str2Str, DiG, ConfDiff, plus new bioRxiv entries every month. Every paper benchmarks its own model on its own systems with its own metrics; there is no neutral, reproducible cross-model comparison. That gap is real and undersupplied.

PREMVAL is the deliberately narrowed, shippable cut of an earlier maximalist NeurIPS-style plan (8 methods, 6 system classes, back-calculated NMR/SAXS, MSM kinetics) — that scope is why it never shipped. This is the WIP=1 version: ATLAS-first, a fixed metric panel, **~$0 compute and zero GPU**, a static leaderboard — sized in weeks, shippable as a public portfolio piece (GitHub repo + pip package + live leaderboard site).

**The differentiator** is **per-model contamination labels**. AlphaFlow-MD trains directly on ATLAS; ANewSampling treats ATLAS as held-out; BioEmu is uncertain. No individual paper surfaces this. A neutral portal that tags each model in-distribution / test / uncertain shows something the field currently hides.

**Intended outcome:** a public, citable, reproducible comparison portal that a company in this space (e.g. Achira) would notice, shipped without a research-grade compute budget.

## Discipline constraints

- **WIP = 1.** This is the single active build. The `brainstorming/` repo is frozen — new ideas get one line in a backlog, no new plan docs.
- **Ship-or-kill, 2-week cadence.** Each milestone either lands or is explicitly re-scoped — no zombie milestones.
- **The prior-art pass is design research, not a kill gate.** Looking at ProteinBench and any existing ensemble-generator leaderboards is purely to learn what makes a comparison *site* good and where existing ones fall short, so PREMVAL is a better site. It is **not** a uniqueness test. The project proceeds regardless of what already exists.

## Scope

**In scope (v1):**
- Reference data: **ATLAS** (~30-target curated subset) + **PED IDP ensembles** (~10 curated uncontaminated entries)
- Metric panel ported wholesale from AlphaFlow's evaluation scripts + two cheap additions — see Metric Panel below
- Per-model contamination labels across all datasets
- pip-installable CPU-only scoring library + CLI (`premval`)
- One defined ensemble submission format (multi-model PDB, fixed 250 frames)
- Static GitHub Pages leaderboard reading committed results JSON
- CASP-style PR submission process
- Seeded **for $0, no GPU** by ingesting AlphaFlow's and ESMFlow's already-published HuggingFace outputs

**Out of scope (v2+):**
- mdCATH (3.61 TB) and post-2024 PDB multi-state track
- TICA free-energy surfaces / BioEmu-style thermodynamic metrics (need kinetics-resolved reference MD)
- Back-calculated experimental observables (NMR/SAXS)
- Live GPU serving; Docker "drop your model in" contract (Docker is the v2 submission contract)
- DESRES fast-folders, protein-ligand, allosteric system classes

## Architecture

Single data flow, no services:

```
reference datasets (ATLAS, PED)
   │  download once, cache locally
   ▼
reference observables  ──── precomputed per target, cached
   │
   │   submitted ensemble (one multi-model PDB per target) ──┐
   ▼                                                         ▼
        scoring library  (pure CPU, pure functions)
   │   score(ensemble, target_id) -> {metric: value, ...}
   ▼
results/<model>.json   ── committed to the repo (transparent, diffable)
   │
   ▼
docs/  static leaderboard  ── GitHub Pages reads results/*.json + contamination_labels.yaml
```

Nothing runs live. v1 needs **no GPU at all** — both seed models are already published; everything else is CPU post-hoc analysis on existing coordinates.

## Ensemble submission format

A submission is a directory (or archive) containing one coordinate file per target plus a manifest.

**Canonical coordinate file** — `<target_id>.pdb`: a multi-model all-atom PDB, one MODEL per sampled conformation. This is exactly what AlphaFlow/ESMFlow emit (`protein.prots_to_pdb`) and what PED uses, so it is mdtraj-loadable with no conversion.

**Fixed ensemble size: 250 frames per target.** AlphaFlow's published outputs are 250 structures/target and the AlphaFlow README is explicit that results are *not comparable across different sample counts*. The scorer subsamples larger submissions to 250 (fixed seed) and rejects submissions with fewer.

**`manifest.json`** (one per submission):
- `model`: string key, must exist in `contamination_labels.yaml`
- `model_version`, `submitted_by`, `date`
- `all_atom`: bool — if false, the physical-validity metric is reported as `null`
- `targets`: list of target IDs included

The scorer aligns each submission to the reference topology by residue index on CA atoms (same approach AlphaFlow's `analyze_ensembles.py` uses).

## Metric panel

**Strategy: port AlphaFlow's evaluation wholesale, add two cheap metrics.** AlphaFlow's `scripts/analyze_ensembles.py` (per-target compute → `out.pkl`) and `scripts/print_analysis.py` (aggregation → table) already implement the de-facto-standard panel that ConfDiff and ANewSampling also adopt. Porting it directly means the M1 port-fidelity check is straightforward and we inherit several metrics "for free." Its deps (`mdtraj`, `sklearn`, `numpy`, `scipy`, `pandas`) match our stack. Note its Wasserstein is computed via `scipy.optimize.linear_sum_assignment` — **no `POT` dependency needed.**

Key functions to port from `analyze_ensembles.py`: `get_pca`, `get_rmsds`, `get_wasserstein`, `get_mean_covar` + `sqrtm` (RMWD), `condense_sidechain_sasas`, `sasa_mi`, `main(name)` with inner `get_emd`. Aggregation/threshold logic (weak vs transient contacts off `crystal_distmat` + contact-probability thresholds) is in `print_analysis.py`.

v1 leaderboard surfaces 6 columns:

| # | Metric | Source | Needs reference? | Notes |
|---|---|---|---|---|
| 1 | RMSF Pearson correlation (per-target + global) | ported | Yes | The single most-reported number in the field. `mdtraj.rmsf`. |
| 2 | RMWD (root-mean Wasserstein distance) | ported | Yes | AlphaFlow's headline distributional metric (`get_mean_covar` + `sqrtm`). |
| 3 | MD-PCA Wasserstein-2 | ported | Yes | PCA fit on reference Cα coords, project both ensembles, EMD via `linear_sum_assignment`. |
| 4 | Weak/transient contact Jaccard | ported | Yes | Contacts = `distmat < 0.8 nm`; weak vs transient by frequency thresholds. Separates collapsed models. |
| 5 | Rg distribution distance (Wasserstein-1) | **added** | Yes | Trivial: `mdtraj.compute_rg` + `scipy.stats.wasserstein_distance`. |
| 6 | Physical validity (clash score + bond-geometry outliers) | **added** | No | All-atom only; `null` if backbone-only. Catches broken outputs. Not in AlphaFlow's eval. |

The port also yields Joint PCA W2, exposed-residue Jaccard, and pairwise RMSD at no extra cost — kept available, surfaced as secondary columns or a drill-down later. None of the panel needs additional MD or GPU.

## Datasets

### ATLAS (~30-target curated subset)
- Source: `dsimb.inserm.fr/ATLAS`. Standardized 3 × 100 ns all-atom MD (CHARMM36m) per protein.
- **Download**: REST API `https://www.dsimb.inserm.fr/ATLAS/api/{dataset}/{archive_type}/{pdb_chain}`, or direct HTTP `https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pdb_chain}/{filename}.zip`, or bulk `download_ATLAS.py` + `aria2c`. Protein IDs are `{pdb_code}_{chain}` lowercase (e.g. `1tag_A`). Archive types: `analysis`, `protein`, `total`, `metadata`.
- **Use the protein-only "MDs" tier** (~700 MB / 10,000 frames for a ~300 aa protein) — skip the 5.3 GB solvated archive. ~30 targets ≈ ~20 GB total.
- Each archive: GROMACS topology PDB + XTC, 3 replicas. Reference loader = download + concat the 3 replica XTCs against the topology PDB (AlphaFlow's `prep_atlas.py` has the concat pattern).
- **Subset selection**: per-protein `metadata.json` + TSVs expose length, organism, fold class, and ECOD/CATH/SCOPe annotations — enough to pick a structurally diverse ~30 spanning fold classes and ~50–300 residues. Committed as `data/atlas_subset.txt`.
- AlphaFlow's `splits/atlas_test.csv` (82 targets; columns `name,seqres,release_date,msa_id,seqlen`) is the pool to curate from — its `release_date` column also feeds temporal contamination labeling.
- **License: CC-BY-NC 4.0.** Precomputed reference metrics are a redistributable adaptation under the same non-commercial terms. One-line note in the repo; matters only if a company wants to fold it into a commercial product.

### PED IDP track (~10 curated entries)
- Source: Protein Ensemble Database (`proteinensemble.org`). 461 entries / 538 ensembles. Genuinely uncontaminated — IDPs are weak in every model's training set.
- **Format**: multi-model PDB (mdtraj-readable, same loader path as AlphaFlow outputs) + optional TSV of per-conformer weights. Conformer count varies per entry (read from MODEL count).
- **API**: `https://proteinensemble.org/api` (Django REST / OpenAPI) — list + selective per-entry download of metadata + PDB ensembles.
- **Curation lever**: origin metadata is queryable (experimental NMR/SAXS/FRET/EPR/CD vs computational MD/MC vs ML-generated). Filter to a single consistent origin class (e.g. experimental NMR/SAXS, excluding ML-generated) for a clean ~10-entry slice. Committed as `data/ped_subset.txt`.
- **License: CC BY** — derived reference data redistributable with attribution, no NC restriction (cleaner than ATLAS).

### Contamination labels — `data/contamination_labels.yaml`
Schema: `model -> dataset -> {status, basis}` where status is `in_distribution | test | uncertain`.
```yaml
alphaflow-md:
  atlas:  {status: in_distribution, basis: "trained on ATLAS train split"}
  ped:    {status: test,            basis: "no IDP MD in training"}
esmflow-md:
  atlas:  {status: in_distribution, basis: "trained on ATLAS train split"}
anewsampling:
  atlas:  {status: test,            basis: "paper treats ATLAS monomers as held-out"}
bioemu:
  atlas:  {status: uncertain,       basis: "broad AFDB+MD corpus; overlap unclear"}
```

## Reuse (do not reimplement)

- **AlphaFlow eval scripts** (`github.com/bjing2016/alphaflow`, `scripts/analyze_ensembles.py` + `scripts/print_analysis.py`): port directly as the metric module.
- **AlphaFlow + ESMFlow published outputs** (HuggingFace `bjing-mit/alphaflow`, `samples/` dir — e.g. `.../resolve/main/samples/alphaflow_md_base_202402.zip`): 250 structures/target, 82 ATLAS test targets, multi-model PDB. Seeds the leaderboard for **$0, no GPU**.
- **AlphaFlow `download_atlas.sh` / `prep_atlas.py`**: ATLAS download + replica-concat pattern.
- **ATLAS** (`dsimb.inserm.fr/ATLAS`) and **PED** (`proteinensemble.org/api`): reference datasets.
- **mdtraj** (trajectory IO, RMSF, Rg, contacts, SASA, superposition), **scipy** (`optimize.linear_sum_assignment`, `stats.wasserstein_distance`, `linalg.sqrtm`), **scikit-learn** (PCA), **numpy**, **pandas**.

## Repo structure

This repo at `~/Projects/premval/`:

```
premval/
├── pyproject.toml              # pip-installable, CPU-only deps
├── README.md
├── PLAN.md                     # this file
├── SUBMISSIONS.md              # CASP-style PR submission instructions + format spec
├── src/premval/
│   ├── io.py                   # submission format read/write, manifest parsing, 250-frame subsample
│   ├── topology.py             # align submission to reference topology by residue index
│   ├── metrics/
│   │   ├── alphaflow_port.py   # ported analyze_ensembles.py + print_analysis.py (metrics 1-4 + bonus)
│   │   ├── rg.py               # metric 5 (added)
│   │   └── validity.py         # metric 6 (added)
│   ├── datasets/
│   │   ├── atlas.py            # REST/HTTP download + 3-replica concat → reference ensemble
│   │   ├── ped.py              # API download → reference ensemble
│   │   └── references.py       # precompute + cache reference observables
│   ├── scoring.py              # score(ensemble, target_id) -> metric dict
│   └── cli.py                  # `premval score|prepare-refs|fetch`
├── data/
│   ├── atlas_subset.txt
│   ├── ped_subset.txt
│   └── contamination_labels.yaml
├── results/                    # one JSON per model — committed leaderboard data
├── docs/                       # GitHub Pages static leaderboard
│   ├── index.html
│   ├── leaderboard.js          # fetches results/*.json + contamination_labels.yaml
│   └── leaderboard.css
└── tests/
    ├── test_metrics.py
    ├── test_io.py
    └── fixtures/               # tiny example ensembles for CI
```

Dependencies: `mdtraj`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `pyyaml`. CLI via `argparse`. No GPU deps, no `POT`.

## Leaderboard site (`docs/`)

Static HTML + vanilla JS, GitHub Pages, zero build step:
- One table. Rows = (model × dataset). Columns = the 6 metrics + a **contamination badge** (green = test/held-out, yellow = uncertain, red = in-distribution).
- `leaderboard.js` fetches `results/*.json` + `contamination_labels.yaml` at load, renders client-side, sort-by-column.
- Per-target drill-down (and the bonus ported metrics) is a v1.1 nicety, not required for first ship.
- Hosting cost: $0. Maintenance: results are PRs, the site is static.

## Milestones (2-week ship-or-kill)

### M1 — Scoring core, one target end-to-end
- [ ] Prior-art pass: ProteinBench + any ensemble-generator leaderboards — note what good comparison sites do well / poorly (design input, not a gate)
- [ ] Confirm `premval` is free on PyPI (local repo dir already exists); claim the GitHub repo name
- [ ] `pyproject.toml`, package skeleton, CI running `pytest`
- [ ] `io.py` + `topology.py`: multi-model PDB read, manifest parsing, 250-frame subsample, residue-index alignment
- [ ] `datasets/atlas.py`: download + 3-replica concat for ~5 ATLAS targets; `references.py`: precompute + cache their reference observables
- [ ] Download AlphaFlow's published outputs (`alphaflow_md_base_202402.zip`) for those 5 targets
- [ ] Port `analyze_ensembles.py` + `print_analysis.py` → `metrics/alphaflow_port.py` (metrics 1–4)
- [ ] `cli.py` + `scoring.py`: score one AlphaFlow output vs one ATLAS reference
- **Done =** CLI produces a metric JSON on CPU. Sanity gate + port-fidelity check (see Verification) pass.

### M2 — Full ATLAS track + seeded public leaderboard  **[first public ship]**
- [ ] Implement metrics 5–6 (Rg W1, physical validity)
- [ ] Curate + commit `atlas_subset.txt` (~30 targets from `atlas_test.csv`, diverse by fold class/length); precompute all reference observables
- [ ] Ingest full AlphaFlow **and** ESMFlow published outputs from HuggingFace (both $0, no GPU)
- [ ] `contamination_labels.yaml` populated with evidence basis for each model
- [ ] `results/*.json` committed; `docs/` leaderboard deployed on GitHub Pages
- [ ] README with screenshots; CC-BY-NC attribution note for ATLAS
- **Done =** public URL showing >=2 models scored across ~30 ATLAS targets with contamination badges.

### M3 — PED IDP track + open submission process
- [ ] `datasets/ped.py`: download curated PED IDP subset via the API (filter to consistent origin class); precompute references
- [ ] Verify the panel behaves sensibly on disordered ensembles; commit `ped_subset.txt`
- [ ] `SUBMISSIONS.md`: CASP-style PR process + finalized format spec
- [ ] Submission dry-run (see Verification)
- [ ] Optional stretch: self-run a 3rd model (ESMFold mode needs no MSA — easiest) to add a non-pre-published entry
- [ ] Public announcement
- **Done =** PED track live on the leaderboard; an external author could submit by following `SUBMISSIONS.md`.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Running other models reproducibly is the real hard part | High | v1 needs zero self-run: AlphaFlow + ESMFlow outputs are both pre-published; CASP-style submission pushes the burden to model authors |
| Ported metrics don't match published numbers | Medium | M1 port-fidelity gate against AlphaFlow paper numbers before building further; we are porting their exact code, so divergence is a bug to fix |
| ATLAS contamination undermines credibility | Low–med | per-model contamination labels are explicit and front-and-center; the PED IDP track is genuinely clean |
| ATLAS CC-BY-NC limits commercial reuse | Low | one-line attribution + NC note in the repo; does not affect a public benchmark; PED is CC-BY |
| Leaderboard goes stale | Medium | static site = zero maintenance; CASP-style submission keeps it fresh without solo upkeep; some staleness is acceptable for a portfolio piece |
| Scope creep back toward the maximalist plan | Medium | the out-of-scope list is the contract; WIP=1; ship M2 before adding anything |
| PED ensembles heterogeneous in origin | Low–med | curate the v1 PED subset to one consistent origin class via the API's origin metadata; documented |

## Verification

- **M1 sanity gate:** score (a) an AlphaFlow ATLAS output and (b) a random-Cartesian-perturbation null model against the same ATLAS reference target. The AlphaFlow ensemble must beat the null on RMSF correlation and MD-PCA W2. If not, the metric implementation is wrong.
- **M1 port-fidelity check:** for >=3 ATLAS targets, RMSF correlation + MD PCA W2 computed by PREMVAL on AlphaFlow's published outputs must match the AlphaFlow paper's reported numbers within rounding.
- **End-to-end:** `pip install` in a fresh venv → run the CLI on a fixture ensemble → metric JSON produced on CPU, no GPU.
- **Leaderboard:** the static site renders committed `results/*.json` and shows the contamination badge per model × dataset row.
- **Submission dry-run:** follow `SUBMISSIONS.md` as an external author — produce a 250-frame multi-model PDB, score it, open a PR against a test branch.

## Concrete next steps

1. Prior-art pass (design input, ~30 min): ProteinBench, any HF ensemble leaderboards.
2. Confirm `premval` PyPI name; create `pyproject.toml`, package skeleton, CI.
3. Clone AlphaFlow; read `scripts/analyze_ensembles.py` + `scripts/print_analysis.py`; start the port into `src/premval/metrics/alphaflow_port.py`.
4. Download 5 ATLAS targets (protein-only MD tier) + the matching slice of `alphaflow_md_base_202402.zip`.
5. Implement `io.py` + `topology.py` + the ported metrics 1–4.
6. Wire `cli.py` + `scoring.py`; run the M1 sanity gate and port-fidelity check.

## Open questions

- Exact ~30 ATLAS target IDs and ~10 PED entry IDs — decided during M2/M3 curation against fold-class/origin metadata.
- Physical-validity thresholds (VdW clash cutoff, bond-geometry sigma) — pick standard MolProbity-style values during M2.
- Whether to surface the bonus ported metrics (Joint PCA W2, exposed-residue J, pairwise RMSD) on the v1 leaderboard or hold for the v1.1 drill-down.
