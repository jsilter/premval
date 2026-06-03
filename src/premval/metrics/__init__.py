"""Metric primitives and the v1 PREMVAL scoring panel.

The four panel-level metrics (`rmsf_correlation`, `rmwd`, `md_pca_w2`,
`contact_jaccard`) live in `panel`. They compose lower-level primitives
(`get_mean_covar`, `get_pca`, `get_wasserstein`, `sqrtm`, `get_rmsds`)
that are faithfully ported from AlphaFlow's `analyze_ensembles.py` and
live in `alphaflow_port`. End-to-end orchestration (load, align, slice,
superpose) is in `premval.scoring`.
"""

from premval.metrics.alphaflow_port import (
    get_mean_covar,
    get_pca,
    get_rmsds,
    get_wasserstein,
    sqrtm,
)
from premval.metrics.panel import (
    ALPHAFLOW_SEED,
    CONTACT_THRESHOLD_NM,
    contact_jaccard,
    contact_probability,
    md_pca_w2,
    rmsf_correlation,
    rmwd,
)

__all__ = [
    "ALPHAFLOW_SEED",
    "CONTACT_THRESHOLD_NM",
    "contact_jaccard",
    "contact_probability",
    "get_mean_covar",
    "get_pca",
    "get_rmsds",
    "get_wasserstein",
    "md_pca_w2",
    "rmsf_correlation",
    "rmwd",
    "sqrtm",
]
