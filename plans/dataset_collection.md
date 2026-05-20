# Plan: Nanobody–Antigen MD Dataset — De-risked Pilot (M0 + M1)

## Context

PREMVAL benchmarks protein conformational-ensemble generators against
reference MD. Its only built reference path today is **ATLAS** (3×100 ns
CHARMM36m MD per chain), loaded as per-chain zip bundles. AlphaFlow/ESMFlow
train *on* ATLAS, so ATLAS is `in_distribution` for them — the project's
headline differentiator (`plans/PLAN.md`) is per-model contamination labels,
and it lacks a reference that is cleanly held-out for *every* model.

This plan **generates a new MD dataset** of **nanobody–antigen complexes** —
freshly simulated, contamination-clean by construction (every system is a PDB
deposition postdating model training cutoffs). The dataset is the deliverable.
Two downstream uses are intended, but their consumers are future work:

1. **Ensemble reference** (positives): a held-out, `test`-labeled MD reference
   for **complex-capable** ensemble generators. *Note:* the four models PREMVAL
   tracks today (AlphaFlow/ESMFlow/BioEmu/ConfDiff) are **monomer** generators
   and cannot consume a 2-chain complex reference — per the user, applicability
   to existing models is not the goal; we are building the dataset itself.
2. **Binding-discrimination** (positives vs. cognate-swap negatives): a
   **model-agnostic** track that scores whether *interface dynamics* (contact
   persistence, interface RMSD, buried SASA) separate real complexes from
   mismatched pairs. This needs no generator and no leaderboard infra.

**This iteration is a de-risked pilot: M0 (curation + contamination proof) and
M1 (3 positives + 3 negatives, end-to-end).** The full ~100–160-system batch,
the contamination-label wiring, and the public leaderboard are explicitly
deferred until M1 validates (a) the bundle/ingest path, (b) a visible
discrimination signal, and (c) the real per-system GPU cost.

### Why a pilot, and what it de-risks

- **Repo discipline.** `plans/PLAN.md` is an explicit WIP=1, $0-compute,
  no-GPU, "ship M2 before adding anything" build. The leaderboard,
  `contamination_labels.yaml`, `results/`, and `docs/` it describes are **not
  built yet** (the tree is mid-build; `cli.py`/`data/__init__.py` import
  `published.py`/`samples.py` not on disk). A multi-$k GPU dataset is a v2+
  effort; the pilot proves the idea without contradicting that discipline or
  blocking on unbuilt infra.
- **Signal risk.** It is unproven that 100 ns surfaces interface
  destabilization in a mismatched complex. M1 measures this on 3+3 before any
  bulk spend.

## Scope decisions (this iteration)

- **Systems**: nanobody (VHH)–antigen complexes from SAbDab-nano, cross-checked
  against SNAC-DB for non-redundant/epitope clustering.
- **Pilot size**: **3 positives + 3 negatives** (6 MD systems). Full set
  (~50–80 positives + comparable negatives) is **deferred** to a follow-up.
- **Negatives**: **derangement swap** — a random permutation with no fixed
  points pairs each nanobody with *another* nanobody's antigen. Positives and
  negatives then share an identical antigen multiset, so a discriminator cannot
  cheat on antigen family/size/composition. Do **not** bias swaps toward
  dissimilar antigens (that reintroduces the confound). Record the full mapping
  + seed. For 3 systems this is a single 3-cycle.
- **Protocol**: match ATLAS exactly (CHARMM36m / TIP3P / 150 mM NaCl /
  3×100 ns, frames every 10 ps) so bundles ingest with **zero** pipeline
  changes and are comparable to ATLAS.
- **Compute**: cloud GPUs (RTX 4090 community / spot preferred for the pilot).

## The reuse boundary (the linchpin — verified)

`load_chain_trajectory` (`src/premval/data/atlas.py:209`) and
`load_reference_observables` (`src/premval/data/references.py:147`) **both
already accept `(chain, kind, cache_dir)`** and resolve paths as:

- bundle: `bundle_path(cache_dir, kind, chain)` → `{cache_dir}/{kind}/{chain}.zip`
- refs:   `cache_path(chain, kind, cache_dir)` → `{cache_dir}/references/{kind}/{chain}.npz`

A bundle is a zip named `{id}.zip` containing `{id}.pdb` (topology only, never a
frame) + `{id}_R1.xtc`, `{id}_R2.xtc`, `{id}_R3.xtc` (atlas.py:249-265).

**Therefore the pilot needs ZERO changes to PREMVAL source.** `--kind` is
argparse-validated against `("analysis","protein","total")` (atlas.py:36), so
we **reuse `kind="analysis"` with a dedicated `--cache-dir`** (e.g.
`~/.cache/premval/nanobody/`). Pipeline emits
`{nb_root}/analysis/{id}.zip`; then:

- `premval prepare-refs --chains <ids> --kind analysis --cache-dir <nb_root>`
  computes/caches reference observables (cli.py:180-193) — unchanged.
- `premval score --chain <id> --cache-dir <nb_root>` scores a submission
  (cli.py:248) — unchanged. (Used only by the future ensemble track.)
- Discrimination metrics call `load_chain_trajectory(id, kind="analysis",
  cache_dir=nb_root)` directly — unchanged.

```
 GPU side (new, pipelines/nanobody_md/)        CPU side (PREMVAL — reused, no source change)
 ───────────────────────────────────────       ───────────────────────────────────────────
 curate.py ──► nb_targets.csv (3 positives)
      │  derangement (seed)
      ▼
 make_negatives.py ──► nb_negatives.csv + co-folded start poses (3)
      │
      ▼  CHARMM36m / TIP3P / 3×100 ns  (ATLAS-matched)
 simulate.py ──► {id}.zip = {id}.pdb + {id}_R{1,2,3}.xtc
      │            (written to  {nb_root}/analysis/{id}.zip )
      ▼
   ┌──────────────────────────────┬───────────────────────────────────┐
   ▼                              ▼                                   ▼
 prepare-refs (CLI, reused)    load_chain_trajectory (reused)     score (CLI, reused)
   .npz refs cache             │  by discrimination.py             [future ensemble track,
                                ▼                                    complex-capable models]
                          discrimination.py (NEW, CPU)
                          interface contacts / iRMSD / buried SASA
                                ▼
                          per-system score + AUROC(positives vs negatives)
```

## Pipeline (pilot) — all new code under `pipelines/nanobody_md/`

This dir is **separate from the pip package** (it needs GPU/MD deps PREMVAL
deliberately avoids). The pip package stays CPU-only and untouched.

**Stage 1 — `curate.py` → `nb_targets.csv` (3 positives).**
- Pull VHH–antigen complexes from SAbDab-nano; cross-check SNAC-DB for
  non-redundant CDR3+epitope clustering, one representative per cluster.
- Filters: X-ray, resolution ≤ 2.5 Å; single VHH + one peptide/single-domain
  antigen (drop large cryo-EM assemblies to keep systems ≤~100k atoms).
- **Contamination gate**: PDB release ≥ 2024-01-01. Verify each ID is absent
  from the vendored ATLAS list (`load_val_chains()`, atlas.py:71 — 39 chains)
  and from AlphaFlow `splits/*.csv`.
- Manifest mirrors the **vendored `atlas_val.csv` schema**
  (`name,seqres,release_date,msa_id` — note `atlas_test.csv` does **not** exist
  in this repo) **plus** columns: antigen_type, resolution, cluster_id.

**Stage 2 — `make_negatives.py` → `nb_negatives.csv` (3 negatives).**
- Compute a seeded derangement of the 3 positive antigens over the 3
  nanobodies (a 3-cycle); record mapping + seed.
- Co-fold each mismatched pair with **AlphaFold3 or Boltz-2** (cheap, minutes)
  for a starting pose; HADDOCK/ClusPro is the docking fallback.
- Label each `assumed_negative`, basis = "derangement swap, no experimental
  complex" — a documented assumption, not validated.

**Stage 3 — `simulate.py` (6 systems).**
- ATLAS-matched: CHARMM36m, TIP3P, 150 mM NaCl, triclinic box; minimize (SD),
  NVT 200 ps → NPT 1 ns; production 3×100 ns @ 10 ps/frame. GROMACS
  (CHARMM-GUI batch prep); OpenMM scriptable fallback.
- VHH care: build canonical Cys22–Cys92 **and** any non-canonical CDR1–CDR3
  disulfide as bonded SS; default-strip partial crystallographic glycans
  (record per target).
- Emit `{id}.pdb` + `{id}_R{1,2,3}.xtc` → `{id}.zip` written to
  `{nb_root}/analysis/{id}.zip`.

**Stage 4 — Ingest + observables (PREMVAL, reused, no source change).**
- `premval prepare-refs --chains <6 ids> --kind analysis --cache-dir <nb_root>`.

**Stage 5 — `discrimination.py` (NEW, CPU).**
- For each system, `load_chain_trajectory(id, kind="analysis",
  cache_dir=nb_root)` → mdtraj trajectory of the complex.
- Compute interface-stability features over the trajectory: native(-ish)
  interface-contact persistence, interface RMSD drift, buried-SASA-over-time
  (`mdtraj.shrake_rupley`), COM separation.
- Output a per-system score JSON; the track metric is **AUROC** of each single
  feature (and the joint panel) separating the 3 positives from the 3
  negatives. (3+3 is a sanity signal, not a powered result — sufficient to
  decide whether 100 ns is long enough before the full batch.)

## Cost (pilot)

`cost/system = (300 ns / ns_per_day) × 24 × $/GPU-hr × 1.08`. ~75k-atom complex:

| GPU (typical $/hr)          | ns/day | GPU-hr (3×100ns) | $/system | 6 systems |
|-----------------------------|--------|------------------|----------|-----------|
| RTX 4090 community (~$0.35) | ~400   | ~54              | ~$20     | **~$120** |
| L40S (~$0.79)               | ~310   | ~70              | ~$60     | ~$360     |
| A100 80GB managed (~$1.10)  | ~370   | ~58              | ~$70     | ~$420     |

Co-folding negative start poses is negligible next to MD. The full-batch
estimate (~$3–12k for 100–160 systems) is **deferred** and re-derived from M1's
measured ns/day.

## Critical files

Reuse **unchanged**: `src/premval/data/atlas.py:209-265` (bundle contract +
loader), `src/premval/data/references.py:147-174` (observables),
`src/premval/cli.py:180-262` (`prepare-refs`, `score`),
`src/premval/data/atlas_val.csv` (manifest schema template),
`src/premval/topology.py:25` (`select_matched_ca`, future ensemble track only).

New (all under `pipelines/nanobody_md/`): `curate.py`, `make_negatives.py`,
`simulate.py`, `discrimination.py`, `run_batch.sh`; outputs `nb_targets.csv`,
`nb_negatives.csv` (+ swap mapping/seed/provenance), `results/*.json`.

## Milestones

- **M0 — Curation + contamination proof.** `nb_targets.csv` (3 non-redundant
  VHH–antigen, release ≥2024, ATLAS-disjoint) + `nb_negatives.csv` (derangement
  mapping + seed). **Done** = manifests committed with a per-target
  contamination check and the swap rationale.
- **M1 — Pipeline on 3+3.** Prep + 3×100 ns + bundle for all 6 on a cloud 4090;
  `prepare-refs` computes references with no source change; `discrimination.py`
  shows positives > negatives on interface persistence. **Done** = 6 bundles
  ingest and produce reference `.npz` + a discrimination JSON on CPU with a
  visible separation signal, and measured ns/day updates the full-batch cost.

**Deferred (follow-up plan, gated on M1):** full ~100–160-system batch; building
the leaderboard/`results`/`docs` infra from `plans/PLAN.md`; adding a
`nanobody` row to `contamination_labels.yaml` with `status: test`; and (if/when
complex-capable generators exist) wiring the ensemble track.

## Verification

- **Bundle contract**: for one generated bundle,
  `load_chain_trajectory(id, kind="analysis", cache_dir=nb_root)` returns 3000
  frames (3×1000) and `load_reference_observables(...)` writes a `.npz` — proves
  drop-in compatibility with no scoring-code changes.
- **Contamination audit**: a `curate.py` assertion that every positive PDB ID
  has release ≥ 2024-01-01 and is absent from `load_val_chains()` and AlphaFlow
  splits.
- **Discrimination calibration**: on the 3+3, confirm interface-contact
  persistence / interface RMSD separate positives from cognate-swap negatives;
  **explicitly check whether 100 ns is long enough** to surface destabilization
  — if not, the follow-up adds longer or enhanced sampling for negatives.
- **Cost reconciliation**: record actual ns/day on the rented GPU; update the
  per-system estimate before proposing the full batch.

## Open items / assumptions

- **Negatives are assumed non-binders** (derangement swap), not experimentally
  validated; a swapped pair could cross-react. Mitigate by labeling
  `assumed_negative` and spot-checking high-risk pairs against known
  cross-reactivity data — **not** by biasing swaps (that reintroduces class
  bias).
- 100 ns may be too short to dissociate a docked decoy; M1 decides whether the
  early-destabilization signal suffices.
- Antigens restricted to peptide/single-domain to keep systems ≤~100k atoms.
- Exact non-redundant counts depend on SAbDab-nano + SNAC-DB filters at run
  time; M0 closes this for the 3 pilot systems.
- The ensemble track's eventual consumers are complex-capable generators;
  today's monomer models (AlphaFlow/ESMFlow/BioEmu/ConfDiff) are out of scope
  for it by design.
