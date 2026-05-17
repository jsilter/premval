# Protein Ensemble Evaluation Metrics

Reference inventory of quantitative metrics used to evaluate generative protein
ensemble methods against molecular-dynamics (MD) reference ensembles. The
primary anchor is AlphaFlow (Jing et al., ICML 2024); follow-up methods are
listed where they introduce or emphasize a metric.

The `Status` column reflects whether the metric is implemented in
`src/premval/metrics/`. This file is a living reference; update it when metrics
are added or sources shift.

## A. Flexibility (per-residue / per-pair amplitude of motion)

| Metric | Definition | Status | Primary sources |
|--------|------------|--------|-----------------|
| RMSF Pearson r (per-target) | Pearson r between per-residue Cα RMSF of model vs. MD | shipped (`panel.rmsf_correlation`) | AlphaFlow Table 1; ProteinBench §3.2.3; AlphaFlow-Lit; EPO; EBA |
| RMSF Pearson r (global pooled) | Same correlation pooled across all targets | not yet | AlphaFlow Table 1; EPO |
| Median RMSF | Median over residues of per-residue RMSF; reported alongside MD reference | not yet | AlphaFlow §5; ProteinBench |
| Pairwise Cα RMSD distribution (Pearson r) | Distribution of all-vs-all Cα RMSDs across the ensemble; correlated with MD's distribution | not yet | AlphaFlow Table 1; AlphaFlow-Lit; EPO |
| dRMSD (distance RMSD) | Per-pair distance variance instead of post-superposition coordinate RMSD | not yet | af2_conformations; Lin et al. 2025 review |
| Dynamic Cross-Correlation Map (DCCM) similarity | Correlation between residue-residue motion covariance maps | not yet | AlphaFlow-Lit (2024) |

## B. Distributional accuracy (whole-ensemble distance to MD)

| Metric | Definition | Status | Primary sources |
|--------|------------|--------|-----------------|
| RMWD (total) | Per-atom 2-Wasserstein between Gaussian fits, root-mean | shipped (`panel.rmwd`) | AlphaFlow Eq. 8 |
| RMWD translation component | Contribution from atom-mean displacement | shipped (sub-output of `panel.rmwd`) | AlphaFlow App. B.3 |
| RMWD variance component | Contribution from atom covariance mismatch | shipped (sub-output of `panel.rmwd`) | AlphaFlow App. B.3 |
| MD-PCA W2 | 2-W in PCA basis fit to MD only | shipped (`panel.md_pca_w2`) | AlphaFlow Table 1 |
| Joint PCA W2 | 2-W in PCA basis fit to equal-weight MD ∪ model | not yet | AlphaFlow Table 1 |
| % PC similarity > 0.5 | Fraction of targets whose top PC cosine-sim with MD exceeds 0.5 | not yet | AlphaFlow Table 1; EPO; EBA; ProteinBench |
| Jensen-Shannon divergence on pairwise Cα distances (JS-PwD) | JS over the matrix of all Cα-Cα distances | not yet | ConfDiff; Str2Str; ESMDiff (ICLR 2025) |
| JS on radius of gyration (JS-Rg) | JS between Rg histograms of model vs. MD | not yet | ESMDiff; EPO (fast-folding) |
| JS on TIC-0 (and TIC-0,1) | JS on first one or two time-lagged independent components | not yet | EPO (fast-folding, tetrapeptides); Str2Str |
| JS on torsion angles (backbone / sidechain / all) | JS on φ/ψ/χ histograms | not yet | EPO (tetrapeptides) |
| TICA divergence | Direct divergence on slow collective modes | not yet | EPO |

## C. Ensemble observables (functional / biophysical readouts)

| Metric | Definition | Status | Primary sources |
|--------|------------|--------|-----------------|
| Weak-contacts Jaccard | Crystal-contact pairs that dissociate in >10% of ensemble | shipped (`panel.contact_jaccard`) | AlphaFlow Table 1 |
| Transient-contacts Jaccard | Non-crystal pairs that associate in >10% of ensemble | shipped (`panel.contact_jaccard`) | AlphaFlow Table 1 |
| Exposed-residue Jaccard | Buried residues that expose sidechain in >10% of ensemble | not yet | AlphaFlow Table 1; ProteinBench |
| Exposed-MI matrix Spearman ρ | Spearman ρ of residue-residue mutual-information matrices on exposure state ("exposon" analysis) | not yet | AlphaFlow Table 1 (Porter & Bowman); ProteinBench |
| RMSEcontact | RMSE between per-pair contact probability matrices | not yet | ConfDiff; EPO (fast-folding) |
| Cryptic-pocket recovery rate | Fraction of known cryptic pockets recovered by ensemble | future work (out of ATLAS scope) | BioEmu (Science 2025) |
| Free-energy difference (ΔΔG) error | absolute error of predicted folding ΔG; correlation r vs. mutant ΔΔG | future work (out of ATLAS scope) | BioEmu |

## D. Structural validity / stereochemistry (per-sample realism)

| Metric | Definition | Status | Primary sources |
|--------|------------|--------|-----------------|
| Cα clash rate | Fraction of frames with at least one Cα-Cα distance below threshold | not yet | ProteinBench; ESMDiff (clash-free validity) |
| Peptide-bond break rate | Fraction of frames with backbone N-C bond length outside tolerance | not yet | ProteinBench |
| Ramachandran validity | Fraction of φ/ψ pairs inside allowed region | not yet | common in MD-validation literature; EPO discussion |
| Bond-length / bond-angle RMSE | Deviation from ideal stereochemistry | not yet | physical-feedback alignment literature (EBA, EPO) |
| Per-frame energy (force-field / NN potential) | Mean / distribution of Amber, Rosetta, or MACE energies; agreement with MD energy distribution | not yet | EPO (Energy Preference Optimization); EBA (2025); force-guided diffusion (ICML 2024) |

## E. Reference-comparison and diversity (PDB-style, not MD-anchored)

| Metric | Definition | Status | Primary sources |
|--------|------------|--------|-----------------|
| Cα lDDT precision | Mean lDDT from each prediction to nearest crystal structure | not yet | AlphaFlow Fig. 3 (PDB eval); Lin et al. 2025 |
| Cα lDDT recall | Mean lDDT from each crystal to nearest prediction | not yet | AlphaFlow Fig. 3 |
| Diversity (1 − pairwise lDDT) | Mean dissimilarity between predicted structures | not yet | AlphaFlow Fig. 3 |
| Ensemble TM-score | TM-score variant over the ensemble | not yet | ESMDiff |
| Best-matching RMSD to reference clusters | Closest-frame RMSD vs. each kinetic cluster | not yet | ESMDiff (fast-folding) |

## F. Sampling efficiency and runtime

| Metric | Definition | Status | Primary sources |
|--------|------------|--------|-----------------|
| Wall-clock convergence to equilibrium observables | GPU-time to match MD's converged RMSF / contacts | not yet | AlphaFlow Fig. 4; BioEmu (4-5 orders speedup) |
| GPU-seconds per sample | Inference cost per generated frame | not yet | EBA Table 1; ProteinBench |

## G. Experimental-data agreement (out of scope; for reference)

These are not used by AlphaFlow or its direct deep-learning successors on
ATLAS, but appear in the broader ensemble-validation literature. Listed here
for completeness; activating any of them is gated on adding an experimental
benchmark dataset (e.g. IDP NMR sets, SAXS curves).

| Metric | Definition | Sources |
|--------|------------|---------|
| Back-predicted NMR chemical-shift RMSE | Compare SHIFTX2 / SPARTA+ predictions on ensemble vs. measured | Robustelli et al. (2018); Vögele et al. (PNAS 2025) |
| 3J-coupling agreement | Per-residue scalar coupling vs. NMR | Springer JBNMR (2020) |
| Order parameter S² | Lipari-Szabo S² from ensemble vs. relaxation experiments | classic NMR validation |
| SAXS profile χ² | Crysol / Pepsi-SAXS predicted vs. experimental I(q) | Lindorff-Larsen group; smFRET-NMR-SAXS integrative papers |
| PRE / smFRET distance distributions | Predicted vs. measured pairwise distance distributions | Schuler et al. (JACS 2020, 2021) |

## Source papers

- Jing et al., "AlphaFold Meets Flow Matching for Generating Protein Ensembles," ICML 2024. [arXiv:2402.04845](https://arxiv.org/abs/2402.04845)
- Huang et al., "Improving AlphaFlow for Efficient Protein Ensembles Generation" (AlphaFlow-Lit). [arXiv:2407.12053](https://arxiv.org/abs/2407.12053)
- Wang et al., "ProteinBench: A Holistic Evaluation of Protein Foundation Models." [arXiv:2409.06744](https://arxiv.org/abs/2409.06744)
- Wang et al., "Protein Conformation Generation via Force-Guided SE(3) Diffusion Models" (ConfDiff), ICML 2024. [arXiv:2403.14088](https://arxiv.org/abs/2403.14088)
- Lu et al., "Str2Str: A Score-based Framework for Zero-shot Protein Conformation Sampling," ICLR 2024.
- Lu et al., "Structure Language Models for Protein Conformation Generation" (ESMDiff), ICLR 2025. [arXiv:2410.18403](https://arxiv.org/abs/2410.18403)
- "Aligning Protein Conformation Ensemble Generation with Physical Feedback" (EBA), 2025. [arXiv:2505.24203](https://arxiv.org/abs/2505.24203)
- "EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization," 2025. [arXiv:2511.10165](https://arxiv.org/abs/2511.10165)
- Zheng et al., "Scalable emulation of protein equilibrium ensembles with generative deep learning" (BioEmu), Science 2025. [doi:10.1126/science.adv9817](https://www.science.org/doi/10.1126/science.adv9817)
- Lin et al., "Beyond static structures: protein dynamic conformations modeling in the post-AlphaFold era," Briefings in Bioinformatics 2025. [doi:10.1093/bib/bbaf340](https://academic.oup.com/bib/article/26/4/bbaf340/8202937)
