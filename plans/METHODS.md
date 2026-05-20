# Protein Ensemble Generator Methods

Companion to `PLAN.md` (the benchmark itself) and `METRICS.md` (the metric
panel). This file summarizes the most prominent generative methods PREMVAL
exists to compare. For each: what the method *is* (1-2 paragraphs), the metrics
it reported, and — for PREMVAL's purposes — whether **(a)** precomputed
ensembles are downloadable and **(b)** the implementation is runnable on the
ATLAS val set.

**Provenance.** Everything below was checked against primary sources via web
search (arXiv, journal metadata, GitHub repos, HuggingFace, Zenodo, the papers'
code-availability sections) rather than written from memory. Quantitative
figures are tagged **[verified]** (read from the paper's own table/abstract) or
**[unverified]** (the HTML table did not extract cleanly; confirm against the
PDF before quoting). Availability links were fetched and, where noted, probed
for liveness.

**Availability at a glance:**

| Method | (a) Downloadable ensembles | (b) Runnable on ATLAS | Weights license |
|--------|---------------------------|-----------------------|-----------------|
| AlphaFlow / ESMFlow | **Yes** — HF, multi-model PDB, the paper's ATLAS ensembles | Yes (GPU; AlphaFlow needs MSAs, ESMFlow doesn't) | MIT |
| ConfDiff | No (generate yourself) | Yes — incl. ATLAS-fine-tuned checkpoint | Apache-2.0 |
| Str2Str | No | Yes — zero-shot; no ATLAS harness, write your own loop | MIT |
| ESMDiff | No | Partial — needs **gated** ESM3 weights, non-commercial | EvolutionaryScale NC |
| DiG | No (download link dead) | Effectively No — weights inaccessible | unstated |
| BioEmu | No (but ATLAS benchmark harness exists) | Yes — open weights; outputs .pdb/.xtc | MIT (open) |
| EBA | **Partial** — 250 test samples as multi-model CIF | Yes (A100; needs Protenix base weights) | MIT + Apache-2.0 |
| EPO | No | No — repo is an empty placeholder (no code yet) | MIT (declared) |

**Practical read for PREMVAL:** AlphaFlow/ESMFlow is the cleanest seed (its
published ATLAS ensembles are exactly the multi-model PDBs `PLAN.md` already
plans to ingest). EBA's artifact tarball is the next easiest *download-and-score*
candidate (CIF → PDB conversion, confirm target list). ConfDiff, Str2Str, and
BioEmu are *run-yourself* options with open weights. ESMDiff is gated behind
ESM3's non-commercial license. DiG (dead weight link) and EPO (no code released)
are not usable today.

---

## AlphaFlow / ESMFlow

Jing, Stärk, Jaakkola & Berger, "AlphaFold Meets Flow Matching for Generating
Protein Ensembles," ICML 2024.

- **Paper:** https://arxiv.org/abs/2402.04845
- **GitHub:** https://github.com/bjing2016/alphaflow (MIT; actively maintained)
- **Weights + ensembles:** https://huggingface.co/bjing-mit/alphaflow (`/tree/main/samples`)
- **ATLAS download script:** in-repo `scripts/download_atlas.sh`

**Method.** Repurposes single-state predictors (AlphaFold2 → AlphaFlow; ESMFold
→ ESMFlow) and fine-tunes them under a **flow-matching** objective to turn each
into a sequence-conditioned *generative* model. Training pairs a noised
structure with the network's clean-structure prediction; integrating the learned
flow from a noise prior at inference yields a *distribution*. PDB-trained gives a
better precision/diversity trade-off than AF2 + MSA-subsampling; the
**AlphaFlow-MD** variants are further trained on ATLAS MD (hence the
in-distribution contamination label). A **distilled** variant is much cheaper.

**Metrics & results.** Defines the de-facto ATLAS panel PREMVAL ports.
**[verified, Table 1, AlphaFlow-MD]**:

- per-target RMSF r = **0.85**; global RMSF r = 0.60
- pairwise-RMSD r = 0.48
- RMWD = 2.61 Å (translation 2.28 / variance 1.30)
- MD-PCA W2 = 1.52 Å; joint-PCA W2 = 2.25 Å
- %PC-sim>0.5 = 44%
- weak/transient contact Jaccard 0.62 / 0.41
- exposed-residue Jaccard 0.50; exposed-MI ρ 0.25

MSA-subsampling AF2 baseline is markedly worse (per-target RMSF r ≈ 0.55).

**Availability**

- **(a) Downloadable ensembles — Yes.** HuggingFace `samples/` hosts 14 zips
  (~9 GB) of the ensembles "used for the analyses and results reported in the
  paper," including `alphaflow_md_base_202402.zip` and ESMFlow/PDB/distilled
  variants, multi-model PDB, ~250 structures/target (download via
  `.../resolve/main/samples/<name>.zip`).
- **(b) Runnable on ATLAS — Yes.** Weights on HF; Python 3.9 / CUDA 11 /
  PyTorch 1.12 / OpenFold, GPU required; AlphaFlow needs `.a3m` MSAs, ESMFlow
  does not.
- *Not fully verified:* that each zip is exactly the ATLAS test split at 250
  frames/target (README says "used in the paper").

---

## ConfDiff

Wang et al., "Protein Conformation Generation via Force-Guided SE(3) Diffusion
Models," ICML 2024.

- **Paper:** https://arxiv.org/abs/2403.14088 (PMLR v235, wang24cv)
- **GitHub:** https://github.com/bytedance/ConfDiff (Apache-2.0; static since 2024-09)
- **Weights:** https://huggingface.co/leowang17/ConfDiff (incl. ConfDiff-MD, ATLAS-fine-tuned)

**Method.** An **SE(3) diffusion** model over backbone frames with a
*force-guided* network combined with a mixture of data-based score models
(classifier-free-style guidance) so samples track the Boltzmann distribution.
Base score net is FramePred-like, concatenating **ESMFold** representations;
trained on PDB. Force guidance is the headline contribution (beats energy
guidance, lower-energy structures).

**Metrics & results.**

- **Datasets:** **12 fast-folding proteins** and **BPTI** (5 states), plus an
  ATLAS evaluation.
- **Metrics:** validity, RMSD, RMSF, JS-PwD/TIC/Rg.
- **[verified]** Abstract claim: surpasses prior SOTA, beating Str2Str and
  EigenFold.
- **Exact JS / validity numbers [unverified]** (HTML tables didn't extract).
- *Secondary source:* third-party benchmark (arXiv:2503.05738) cites
  ConfDiff-Open-PDB RMSF r ≈ 0.886 vs AlphaFlow-PDB ≈ 0.891.

**Availability**

- **(a) Downloadable ensembles — No.** README documents *generating* samples;
  no pre-generated ensemble download found.
- **(b) Runnable on ATLAS — Yes.** Weights released (ConfDiff-BASE, -GUIDANCE
  force/energy, and **-MD fine-tuned on ATLAS**); repo includes ATLAS data-prep,
  so the ATLAS path is realistic. Built on ESMFold/OpenFold, GPU required.
- *Not verified:* exact GPU memory, MSA path for the OpenFold-representation
  variant.

---

## Str2Str

Lu, Zhong, Zhang & Tang, "Str2Str: A Score-based Framework for Zero-shot Protein
Conformation Sampling," ICLR 2024.

- **Paper:** https://arxiv.org/abs/2306.03117
- **GitHub:** https://github.com/lujiarui/Str2Str (MIT; actively maintained)
- **Weights:** released via Google Drive (into `data/ckpt`, see README)

**Method.** A **zero-shot** sampler needing no MD data. A roto-translation
equivariant score net is trained on PDB crystal structures with amortized
denoising score matching; at inference (simulated-annealing style) it perturbs
an input structure forward then denoises to emit a new conformation. Two
samplers (SDE, probability-flow). Generalizes across targets zero-shot.

**Metrics & results.**

- **Metrics:** validity (Val-Clash, Val-Bond); fidelity JS-PwD/TIC/Rg; diversity
  MAE-RMSD/MAE-TM vs full MD.
- **Datasets:** 12 fast-folders + BPTI.
- **[verified, Table 1]** Str2Str(SDE) vs EigenFold vs idpGAN:
  - JS-PwD **0.348** / 0.530 / 0.480
  - JS-TIC 0.400 / 0.497 / 0.517
  - JS-Rg 0.365 / 0.666 / 0.661
  - Val-Bond 0.982 / 0.874 / 0.032
- **[verified, Table 2]** fidelity comparable to 100 µs MD in ≈510 GPU-seconds
  vs >160 GPU-days.

**Availability**

- **(a) Downloadable ensembles — No.** Only checkpoint + eval scripts, no
  precomputed PDBs.
- **(b) Runnable on ATLAS — Yes.** Weights released; PyTorch+Lightning+Hydra,
  GPU expected. **ATLAS never used** in repo/paper (fast-folders + BPTI only),
  but running on ATLAS targets is straightforward in principle: feed each ATLAS
  reference structure as input and sample. No ATLAS harness exists — you'd write
  the input-prep/sampling loop.
- *Not verified:* exact input-dir schema, sample-count flag, VRAM.

---

## ESMDiff

Lu et al., "Structure Language Models for Protein Conformation Generation"
(ESMDiff), ICLR 2025.

- **Paper:** https://arxiv.org/abs/2410.18403
- **GitHub:** https://github.com/lujiarui/esmdiff (EvolutionaryScale Cambrian **Non-Commercial**)
- **Weights:** ESMDiff checkpoint via Google Drive; **requires gated** ESM3 weights at https://huggingface.co/EvolutionaryScale/esm3

**Method.** Introduces **Structure Language Modeling**: encode 3D structures into
a compact *discrete* latent via a discrete VAE, then do conditional language
modeling over those tokens. **ESMDiff** is the flagship — a BERT-like structure
LM fine-tuned from **ESM3 with masked (discrete) diffusion** plus a seq→str
inductive bias. Avoids slow continuous-coordinate 3D diffusion.

**Metrics & results.**

- **Metrics:** JS on PwD/TIC/Rg; clash-free validity; ensemble TM-score and RMSD
  vs kinetic clusters; RMSF correlation; IDP MAE.
- **Datasets:** BPTI, conformational-change pairs (77 fold-switching + 90
  apo/holo), 114 PED IDPs.
- **[unverified — re-check against PDF]** BPTI: ESMDiff(DDPM) JS-PwD 0.372,
  JS-TIC 0.420, JS-Rg 0.439, validity 0.940, TM-ens 0.843; apo/holo RMSF
  correlation 0.420.
- Reported **20–100×** speedup over diffusion methods like AlphaFlow.

**Availability**

- **(a) Downloadable ensembles — No.** Only the checkpoint + a BPTI example; no
  precomputed outputs.
- **(b) Runnable on ATLAS — Partial.** Checkpoint released, but inference
  **requires gated ESM3 weights** (HF account + accept non-commercial license);
  the whole stack is non-commercial. PyTorch+Lightning+Hydra, sizable GPU. ATLAS
  not used in the paper; runnable on ATLAS only after clearing ESM3 gating and
  writing your own input loop.

---

## Distributional Graphormer (DiG)

**Citation corrected:** Zheng, He, Liu, Shi, Lu, et al. (Microsoft Research
AI4Science), "Predicting equilibrium distributions for molecular systems with
deep learning," **Nature Machine Intelligence 6, 558–567 (2024)** — *not* Nature
Computational Science.

- **Paper (journal):** https://www.nature.com/articles/s42256-024-00837-3
- **Paper (preprint):** https://arxiv.org/abs/2306.05445
- **Website:** https://distributionalgraphormer.github.io (interactive demo only)
- **GitHub:** https://github.com/microsoft/Graphormer/tree/main/distributional_graphormer
- **Weights/data:** *(dead)* Azure SAS link expired 2025-04-04 — see issue #199

**Method.** A diffusion-based generative framework (Graphormer backbone) that
predicts the *equilibrium distribution* of molecular structures conditioned on a
system descriptor (graph or sequence), inspired by thermodynamic annealing.
Broad scope — protein conformations, ligand poses, catalyst–adsorbate, and
property-guided generation — rather than ATLAS-specific.

**Metrics & results.**

- **Scope:** diverse-conformation generation, equilibrium
  state-density/population estimation, and free-energy projections.
- **No specific numbers verified** (open abstract is qualitative; Nature MI full
  text paywalled).
- Sits *adjacent* to PREMVAL's ATLAS panel.

**Availability**

- **(a) Downloadable ensembles — No.** No precomputed protein ensembles; the
  only checkpoint/data host was an Azure SAS URL that expired 2025-04-04 (GitHub
  issue #199 confirms a 409 "public access not permitted").
- **(b) Runnable on ATLAS — Effectively No.** Inference code is public but the
  trained weights are inaccessible (no HuggingFace/Zenodo mirror found); you'd
  have to obtain weights from the authors.

---

## BioEmu

**Citation corrected:** lead/senior authors are **Lewis, Hempel, Jiménez-Luna,
… Clementi & Noé** (Microsoft Research), not Zheng.

- **Paper:** https://www.science.org/doi/10.1126/science.adv9817 (Science 389(6761), eadv9817, 2025)
- **Preprint:** bioRxiv 2024.12.05.626885
- **GitHub:** https://github.com/microsoft/bioemu (MIT; actively maintained) + https://github.com/microsoft/bioemu-benchmarks
- **Weights:** https://huggingface.co/microsoft/bioemu (MIT, open; bioemu-v1.0/1.1/1.2)

**Method.** A large-scale generative *equilibrium-ensemble emulator* producing
thousands of independent structures/hour on one GPU. Trained by integrating
**>200 ms** of MD, static structures (PDB/AFDB), and experimental
protein-stability data, so it jointly models structural ensembles *and*
thermodynamics — the scaled successor to the DiG line.

**Metrics & results.**

- **Readouts** (beyond the ATLAS panel): folding ΔG and mutation ΔΔG accuracy,
  cryptic-pocket / conformational-change recovery, speedup vs MD.
- **[verified verbatim in the Science/PubMed abstract]** relative free energies
  to **~1 kcal/mol**; **>200 ms** MD integrated.
- **[secondary]** stability errors <1 kcal/mol, correlation >0.6 for ΔG/ΔΔG,
  pocket/change success ~55–90%, 4–5 OOM speedup.
- Mostly **out of PREMVAL v1 scope**; ATLAS contamination *uncertain*.

**Availability**

- **(a) Downloadable ensembles — No.** No precomputed sample ensembles, **but**
  `bioemu-benchmarks` ships an `md_emulation` benchmark over **50 ATLAS cases**
  (ATLAS is *not* in BioEmu training) with projection matrices + metrics code —
  you generate the samples yourself.
- **(b) Runnable on ATLAS — Yes.** MIT, open weights auto-downloaded from HF;
  Linux `pip install bioemu` (Python 3.10+, `[cuda]`), single GPU (A100-80GB
  benchmarked, ~4 min/100 res). Outputs backbone frames; side-chain
  reconstruction gives all-atom **`.pdb` + `.xtc`** (may need xtc→multi-model PDB
  conversion). No blockers to running on ATLAS.

---

## Physical-feedback alignment: EBA and EPO

Not new generators but **alignment / fine-tuning** layers on top of existing
ensemble models, borrowing preference optimization (DPO lineage) from LLM
alignment with a physical *energy* signal as the reward.

### EBA — Energy-based Alignment

Lu, Chen, Lu, Lozano, Chenthamarakshan, Das & Tang, ICML 2025.

- **Paper:** https://arxiv.org/abs/2505.24203 (PMLR v267, lu25b)
- **GitHub:** https://github.com/lujiarui/eba (MIT; maintained)
- **Weights:** https://storage.googleapis.com/project_icml25_eba/release.pt (~3.1 GB, live)
- **Data download:** https://storage.googleapis.com/project_icml25_eba/artifacts.tar.gz (~1.29 GB, live)

**Method.** Generalized preference-optimization: sample K conformations, minimize
cross-entropy between Boltzmann-weighted target energies and model
log-probabilities; **DPO is the special case K=2, β→∞ [verified]**. Fine-tunes
a **Protenix** (ByteDance AF3 reproduction) base. Evaluated on the **ATLAS**
panel. Verified claim: SOTA on the MD ensemble benchmark; **specific numbers
[unverified]**.

**Availability**

- **(a) Downloadable ensembles — Partial.** The artifacts tarball holds **250
  test samples as multi-model CIF** (convert to PDB; strongly implied to be the
  ATLAS test targets, but the README doesn't enumerate them — confirm by
  inspecting).
- **(b) Runnable on ATLAS — Yes.** Full pipeline (install + cutlass build, data
  prep, inference, SFT/EBA finetuning, `analyze_ensembles.py`); EBA checkpoint
  live. Needs the **Protenix base weights** (`model_v0.2.0.pt`) + AF3 release
  MSAs separately, plus raw ATLAS. A100-class GPU.

### EPO — Energy Preference Optimization

Sun, Ren, Chen, Han, Liu & Ye, AAAI 2026.

- **Paper:** https://arxiv.org/abs/2511.10165 (AAAI v40(2), pp. 1060–1068)
- **GitHub:** https://github.com/sunyuancheng/EPO (MIT declared; **empty placeholder — no code yet**)

**Method.** An *online refinement* method turning a pretrained generator (base:
**MDGen [verified]**) into an energy-aware sampler without extra MD, via SDE
sampling + a **list-wise** preference-optimization energy ranking and an
upper-bound approximation to the continuous-time trajectory probability. Energy
reward via the differentiable Madrax force field ([unverified]). Benchmarks:
**tetrapeptides, ATLAS, fast-folding**. Verified claim: "new SOTA on nine
metrics"; **specific numbers [unverified]**.

**Availability**

- **(a) Downloadable ensembles — No.** No outputs anywhere.
- **(b) Runnable on ATLAS — No.** The official repo is a stub (created
  2025-11-12, never pushed, only LICENSE + one-line README); no code, no
  weights. You'd also need MDGen weights when/if released. (The `lxqpku/EPO`
  repo from search is an unrelated ACL 2025 LLM paper — ignore it.)

---

## Where each sits relative to PREMVAL (contamination)

| Method | Trains on MD/ATLAS? | Likely ATLAS contamination label |
|--------|--------------------|----------------------------------|
| AlphaFlow-MD / ESMFlow-MD | Yes (ATLAS) | in_distribution |
| AlphaFlow / ESMFlow (PDB) | No (PDB only) | test-ish |
| Str2Str | No (zero-shot, PDB) | test |
| ConfDiff | PDB base; ATLAS-FT variant exists | depends on variant |
| ESMDiff | Fine-tunes ESM3 | uncertain |
| DiG | sim-free + data, broad corpus | n/a (not ATLAS-centric) |
| BioEmu | Yes (huge MD+AFDB+stability corpus) | uncertain (ATLAS not in training per bioemu-benchmarks) |
| EBA | Fine-tunes Protenix on ATLAS | inherits / in_distribution |
| EPO | Fine-tunes MDGen | inherits |

The MD-trained models (AlphaFlow-MD, EBA) score best on ATLAS partly because
ATLAS is in their training set — the contamination story `PLAN.md` calls the
differentiator.

---

## To lock down before quoting numbers publicly

HTML scraping did not cleanly return result tables for **ConfDiff, ESMDiff, EBA,
EPO** — pull the PDFs before any number goes on the leaderboard:

- ConfDiff: https://arxiv.org/pdf/2403.14088
- ESMDiff: https://arxiv.org/pdf/2410.18403
- EBA: https://arxiv.org/pdf/2505.24203
- EPO: https://arxiv.org/pdf/2511.10165
- BioEmu full text (paywalled): Science eadv9817 — confirm ΔG/ΔΔG correlation
  and pocket-recovery figures.

Also confirm-by-inspection: which exact targets EBA's 250-sample CIF artifact
covers, and whether AlphaFlow's HF zips are the 82-target ATLAS test split at
250 frames each.
