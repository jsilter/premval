"""ATLAS dataset access for premval.

Exposes the AlphaFlow `val` split of the ATLAS MD database (39 PDB chains)
and a downloader that pulls per-chain trajectory bundles from the ATLAS
HTTP API into a local cache.

The 39-chain list is vendored from
`bjing2016/alphaflow:splits/atlas_val.csv` so we have a stable reference
without a runtime dependency on GitHub.
"""

from premval.data.atlas import (
    ATLAS_KINDS,
    ATLAS_REPLICAS,
    AtlasKind,
    default_cache_dir,
    fetch_val_split,
    load_chain_trajectory,
    load_ensemble_pdb_bytes,
    load_topology_bytes,
    load_val_chains,
)
from premval.data.references import ReferenceObservables, load_reference_observables

__all__ = [
    "ATLAS_KINDS",
    "ATLAS_REPLICAS",
    "AtlasKind",
    "ReferenceObservables",
    "default_cache_dir",
    "fetch_val_split",
    "load_chain_trajectory",
    "load_ensemble_pdb_bytes",
    "load_reference_observables",
    "load_topology_bytes",
    "load_val_chains",
]
