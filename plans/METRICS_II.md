# General-Purpose Distribution and Ensemble Metrics

Reference inventory of distribution-comparison and ensemble-evaluation metrics
drawn from probabilistic-programming, Bayesian, and ML / statistics libraries.
Companion to `METRICS.md`, which catalogues protein-anchored metrics from
AlphaFlow and its follow-ups. The point of this file is to give PREMVAL
contributors a vocabulary beyond the protein-specific tradition; everything
below is a candidate for adoption once a featurization is fixed (torsions,
pairwise distances, contact maps, PCA coordinates, learned embeddings).

Scope: only metrics that compare two empirical distributions or two
ensembles. MCMC convergence diagnostics, Bayesian model comparison (LOO,
WAIC), posterior predictive checks, calibration of scalar predictions, and
pure density-estimation log-likelihoods are out of scope; they diagnose a
posterior the user is sampling, which is not PREMVAL's setting.

None of these are shipped in PREMVAL today; the question this file answers is
"what could be added, and where does the implementation live."

## A. Classical two-sample tests (univariate)

| Metric | Definition | Library / call | Notes |
|--------|------------|----------------|-------|
| Kolmogorov-Smirnov (KS) | Max distance between two empirical CDFs | `scipy.stats.ks_2samp` | Common drift baseline; insensitive to tails |
| Anderson-Darling (k-sample) | Weighted ECDF distance, heavier tail weighting | `scipy.stats.anderson_ksamp` | Generally more powerful than KS for small differences |
| Cramér-von Mises (2-sample) | Integrated squared ECDF distance | `scipy.stats.cramervonmises_2samp` | Smoother than KS; symmetric, no parameter |
| Mann-Whitney U | Rank-sum test for stochastic ordering | `scipy.stats.mannwhitneyu` | Tests "is one distribution shifted," not full-distribution equality |
| Epps-Singleton | Characteristic-function-based; works on continuous and discrete data | `scipy.stats.epps_singleton_2samp` | More powerful than KS when distributions differ in shape, not location |
| Wald-Wolfowitz runs | Runs test on sorted pooled samples | `statsmodels` (sandbox runs module) | Older; included for completeness |
| Pearson χ² (binned) | Discrepancy on bin counts | `scipy.stats.chi2_contingency` | For discrete features or pre-binned continuous data |

When to reach for this family: per-feature comparisons of a scalar observable
(per-residue RMSF, per-torsion angle, per-pair distance) where a fixed
multiple-testing correction (Bonferroni, BH-FDR) is acceptable for the
hundreds-to-thousands of features a residue-level ensemble produces.

## B. f-divergences and information-theoretic distances

| Metric | Definition | Library / call | Notes |
|--------|------------|----------------|-------|
| KL divergence (forward) | E_p[log p/q] | `scipy.special.rel_entr` for arrays; `tfp.distributions.kl_divergence` for closed-form between named families | Asymmetric, mode-seeking; requires q > 0 on supp(p) |
| KL divergence (reverse) | E_q[log q/p] | same | Mode-covering; common in variational training |
| Jensen-Shannon (JS) | (1/2)[KL(p‖m) + KL(q‖m)] with m = (p+q)/2 | `scipy.spatial.distance.jensenshannon` | Symmetric, bounded; the form used by ConfDiff / ESMDiff / EPO |
| Hellinger | (1/√2)‖√p − √q‖₂ | hand-rolled on histograms | Symmetric, bounded; well-behaved when densities overlap weakly |
| Total variation (TV) | (1/2)‖p − q‖₁ | hand-rolled on histograms | Probabilistic upper bound on absolute difference of any event probability |
| Pearson χ² | E_q[(p/q − 1)²] | bin-count ratio | f-divergence; tail-sensitive |
| Rényi-α divergence | (1/(α−1)) log E_q[(p/q)^α] | hand-rolled; `tfp.experimental.distributions.RenyiDivergence` | Family containing KL (α → 1) and Hellinger (α = 1/2) |
| Population Stability Index (PSI) | Binned symmetric KL | `evidently.metrics` (DataDrift preset) or hand-rolled | Industry-standard drift thresholds 0.1 / 0.25 |

When to reach for this family: when the question is "how different are these
two distributions" in an information-theoretic sense, and both can be
tractably binned (or both belong to the same named distribution family).
f-divergences require common support; if the predicted ensemble misses an MD
basin entirely, JS and TV are bounded while KL is infinite (which sometimes
is and sometimes is not the desired behavior).

## C. Integral probability metrics (IPMs)

| Metric | Definition | Library / call | Notes |
|--------|------------|----------------|-------|
| Wasserstein-1 (1D) | ∫|F_p − F_q| | `scipy.stats.wasserstein_distance` | Closed-form via sorted ECDFs |
| Wasserstein-2 (1D / Gaussian) | ECDF L² distance; closed form for Gaussians | `scipy.stats.wasserstein_distance` (W1); manual for W2; `tfp` for Gaussians | Already used by PREMVAL's RMWD (Gaussian closed-form per atom) |
| Earth Mover Distance (exact, multi-D) | Optimal-transport cost | `ot.emd2(a, b, M)` | Exact but O(n³ log n); use ≤ a few thousand samples |
| Entropic-regularized OT (Sinkhorn) | EMD + ε · entropy | `ot.sinkhorn2` | Fast and differentiable; biased; use Sinkhorn divergence to debias |
| Sliced Wasserstein | Average W₂ over random 1D projections | `ot.sliced_wasserstein_distance` | Scales to high dim; standard for ML evaluation |
| Max-sliced Wasserstein | Max over a learned 1D direction | `ot.max_sliced_wasserstein_distance` | Adversarial projection; sharper than sliced |
| Gromov-Wasserstein | OT between metric spaces, no shared coords | `ot.gromov.gromov_wasserstein2` | Does not require aligned coordinates; useful across topologies |
| Fused Gromov-Wasserstein | Mix of GW and standard W on attributed graphs | `ot.gromov.fused_gromov_wasserstein2` | Combines structure (GW) with feature (W) costs |
| MMD² (RBF, polynomial, learned kernel) | ‖μ_p − μ_q‖²_H in an RKHS | `alibi_detect.cd.MMDDrift`; `torchdrift.detectors.KernelMMDDriftDetector` | Kernel choice and bandwidth matter; median-heuristic standard |
| Learned-kernel MMD | MMD with a trained deep kernel | `alibi_detect.cd.LearnedKernelDrift` | Higher power on hard cases; needs a hold-out split |
| Energy distance | Special case of MMD with negative-distance kernel | `scipy.stats.energy_distance` (1D); `dcor.energy_distance` (multi-D) | No bandwidth hyperparameter |
| Kernel Stein Discrepancy (KSD) | Stein operator on an RKHS evaluated under q | hand-rolled with autograd; research code | One-sample GoF; needs ∇log p of the target |

When to reach for this family: IPMs handle disjoint supports gracefully
(cost grows continuously as distributions separate, unlike KL which
diverges). For PREMVAL the natural fits are sliced-W (one or two orders of
magnitude cheaper than EMD on coordinate ensembles), Gromov-W (when
comparing ensembles that lack a common alignment), and MMD with
median-heuristic RBF (standard, nonparametric, paired with a permutation
test).

## D. Two-sample tests with type-I error control

| Test | Definition | Library / call | Notes |
|------|------------|----------------|-------|
| MMD permutation test | MMD² with label-shuffle null | `alibi_detect.cd.MMDDrift(p_val=...)` | Standard nonparametric multivariate two-sample test |
| Classifier two-sample test (C2ST) | Train classifier P vs Q; accuracy − 1/2 | `alibi_detect.cd.ClassifierDrift` | Lopez-Paz & Oquab, ICLR 2017; works in any feature space |
| Learned-kernel MMD test | Kernel optimized for test power | `alibi_detect.cd.LearnedKernelDrift` | Liu et al., ICML 2020 |
| Kernel Stein test (1-sample GoF) | KSD with bootstrap null | research code | When the reference is a known density, not samples |
| Relative GoF test | Compare two candidate models to one reference | research code | Bounliphone et al., 2016 |
| Context-aware MMD | MMD conditional on a known covariate | `alibi_detect.cd.ContextMMDDrift` | Detects drift while accounting for known covariate shift |
| Spot-the-diff | Identifies which features drove the drift | `alibi_detect.cd.SpotTheDiffDrift` | Interpretable; not just a p-value |

When to reach for this family: when the leaderboard needs to make a
significance claim ("method X is closer to MD than method Y at p < 0.01"),
not just an ordering claim. All of these are bootstrapping or permutation
tests built around the IPMs in §C; the value-add over §C is the calibrated
null distribution.

## E. Drift detection panels (production-flavored)

| Method | Definition | Library / call | Notes |
|--------|------------|----------------|-------|
| PSI | Binned symmetric KL with industry thresholds | `evidently.metrics` | Default thresholds 0.1 (warn) / 0.25 (alarm) |
| KS-FDR / KS-Bonferroni (multivariate) | Per-feature KS + correction | `alibi_detect.cd.KSDrift` | Aggregates many univariate tests across features |
| Chi² drift (categorical) | Per-feature χ² + correction | `alibi_detect.cd.ChiSquareDrift` | Categorical analog of KS-FDR |
| LSDD | Least-squares density difference | `alibi_detect.cd.LSDDDrift` | Faster than MMD on large reference sets |
| Online MMD | MMD with ERT-calibrated thresholds | `alibi_detect.cd.MMDDriftOnline` | Sequential testing with controlled false-alarm rate |
| Classifier drift (online or batch) | C2ST + drift threshold | `alibi_detect.cd.ClassifierDrift`, `ClassifierDriftOnline` | Same as §D, packaged for production |
| Wasserstein drift | scipy W₁ with threshold | `evidently.metrics` | Default threshold 0.1 |
| Anderson-Darling drift | k-sample AD + threshold | `evidently.metrics` | Stricter than KS for tail differences |

When to reach for this family: alibi-detect, Evidently, and torchdrift are
the "production-packaged" wrappers around metrics already covered in §A-§D.
The value-add for PREMVAL is operational; each detector exposes a default
threshold, a streaming variant, and a well-defined "drift detected" boolean
output suitable for CI gates on regression-test ensembles.

## F. Proper scoring rules (calibration of probabilistic forecasts)

| Score | Definition | Library / call | Notes |
|-------|------------|----------------|-------|
| Log score (Ignorance) | −log p(y_obs) | hand-rolled given density | Strictly proper; the likelihood itself |
| CRPS (univariate) | ∫(F(x) − 1[x ≥ y_obs])² dx | `properscoring.crps_ensemble`; `scoringrules.crps_ensemble` | Generalization of MAE to probabilistic forecasts |
| Energy score (multivariate CRPS) | E‖X − y‖ − (1/2)E‖X − X'‖ | `properscoring.energy_score`; `scoringrules.energy_score` | Multivariate generalization of CRPS |
| Variogram score | Σ_{i,j} w_{ij}(|y_i − y_j|^p − E|X_i − X_j|^p)² | `scoringrules.variogram_score` | Sensitive to dependence structure, not just marginals |
| Brier score | (p − y)² for binary outcome | `sklearn.metrics.brier_score_loss` | For binary observables (e.g., a contact is or isn't formed) |
| Quantile (pinball) loss | Asymmetric L1 at quantile τ | `scoringrules.quantile_score` | Per-quantile calibration |
| Dawid-Sebastiani | (y − μ)²/σ² + log σ² | hand-rolled | First two moments only; cheaper proxy for log score |

When to reach for this family: when the "ground truth" is a single MD frame
(or a small set of frames) rather than a full reference ensemble, and the
model output is an ensemble interpreted as a predictive distribution. Energy
score in particular is the principled multivariate analog of CRPS; it is
what weather and climate ensemble forecasting use for exactly this kind of
comparison.

## G. Generative-model evaluation (image / embedding flavored)

| Metric | Definition | Library / call | Notes |
|--------|------------|----------------|-------|
| FID (Fréchet Inception Distance) | W₂ between Gaussian fits of features | `torchmetrics.image.fid.FrechetInceptionDistance`; `cleanfid` | Sample-size biased; use ≥ 10k samples for stable values |
| KID (Kernel Inception Distance) | MMD² with polynomial kernel on features | `torchmetrics.image.kid.KernelInceptionDistance` | Unbiased; smaller sample sizes OK |
| Inception Score (IS) | KL(p(y|x) ‖ p(y)) over a classifier | `torchmetrics.image.inception.InceptionScore` | Image-specific; weak signal for non-photographic data |
| Precision / recall (Sajjadi 2018, Kynkäänniemi 2019) | Manifold-based fidelity / diversity | `prdc.compute_prdc`; `torch-fidelity` | Separates "are samples realistic" from "do they cover the reference" |
| Density / coverage (Naeem 2020) | Improved precision / recall robust to outliers | `prdc.compute_prdc` | Same call as precision / recall |
| Authenticity (Alaa 2022) | Memorization vs. novelty | research code | Catches training-set leakage |

When to reach for this family: FID, KID, and the precision/recall family
are explicitly built to evaluate generative models in arbitrary feature
spaces; they transfer directly to protein ensembles once an embedding is
chosen (ESM tokens, an lDDT-vector autoencoder, even a fixed PCA basis on
backbone atoms). They are the only family in this document that cleanly
decomposes "fidelity" from "diversity," which the protein-anchored panel in
`METRICS.md` does not.

## Mapping to PREMVAL

Three notes on how to bring these general metrics to bear on the
protein-ensemble setting:

- Feature-space choice is upstream of metric choice. Most IPMs, two-sample
  tests, and drift detectors presuppose a fixed featurization (torsions,
  pairwise Cα distances, contact maps, PCA on Cα coordinates, learned
  embeddings). Adopting any §C-§E metric means committing to a feature
  pipeline; that pipeline is then reusable across the families.
- Sampling-error awareness. ATLAS bounds the available frames per chain;
  every metric in this file is an estimator with its own variance. The
  100-bootstrap protocol that AlphaFlow uses on its panel applies equally to
  anything added from here, with the IPMs in §C requiring particular care
  because their bias scales with sample size.
- Point estimate vs. test framing. PREMVAL today reports point distances
  (RMWD, PCA-W2, contact Jaccards). The §D family lets the leaderboard say
  "method X is significantly closer to MD than method Y" instead of just
  "method X ranks first"; that is usually a more honest framing when
  per-chain variance is large.

## Source links

- [scipy.stats reference](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [POT (Python Optimal Transport)](https://pythonot.github.io/)
- [TensorFlow Probability](https://www.tensorflow.org/probability)
- [alibi-detect (Seldon)](https://docs.seldon.io/projects/alibi-detect/en/stable/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [torchdrift](https://torchdrift.org/)
- [properscoring](https://github.com/properscoring/properscoring)
- [scoringrules (Python)](https://frazane.github.io/scoringrules/)
- [torchmetrics](https://lightning.ai/docs/torchmetrics/)
- [clean-fid](https://github.com/GaParmar/clean-fid)
- [torch-fidelity](https://github.com/toshas/torch-fidelity)
- [prdc (clovaai)](https://github.com/clovaai/generative-evaluation-prdc)
- Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests," ICLR 2017. [arXiv:1610.06545](https://arxiv.org/abs/1610.06545)
- Liu et al., "Learning Deep Kernels for Non-Parametric Two-Sample Tests," ICML 2020. [arXiv:2002.09116](https://arxiv.org/abs/2002.09116)
- Liu et al., "A Kernelized Stein Discrepancy for GoF Tests," ICML 2016. [arXiv:1602.03253](https://arxiv.org/abs/1602.03253)
- Sajjadi et al., "Assessing Generative Models via Precision and Recall," NeurIPS 2018. [arXiv:1806.00035](https://arxiv.org/abs/1806.00035)
- Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models," NeurIPS 2019. [arXiv:1904.06991](https://arxiv.org/abs/1904.06991)
- Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models," ICML 2020. [arXiv:2002.09797](https://arxiv.org/abs/2002.09797)
- Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," JASA 2007.
