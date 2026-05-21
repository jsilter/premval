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
    bundle_path,
    default_cache_dir,
    fetch_val_split,
    load_chain_trajectory,
    load_ensemble_pdb_bytes,
    load_test_chains,
    load_topology_bytes,
    load_val_chains,
)
from premval.data.published import PUBLISHED_SOURCES, PublishedSource, fetch_published
from premval.data.rcsb import (
    assembly_cache_path,
    fetch_assembly_pdb,
    load_assembly_bytes,
)
from premval.data.references import ReferenceObservables, load_reference_observables
from premval.data.samples import (
    aligned_sample_path,
    available_chains,
    available_models,
    default_samples_dir,
    load_aligned_sample_pdb_bytes,
    load_sample_observables,
    load_sample_pdb_bytes,
    sample_observables_path,
    sample_path,
)

__all__ = [
    "ATLAS_KINDS",
    "ATLAS_REPLICAS",
    "PUBLISHED_SOURCES",
    "AtlasKind",
    "PublishedSource",
    "ReferenceObservables",
    "aligned_sample_path",
    "assembly_cache_path",
    "available_chains",
    "available_models",
    "bundle_path",
    "default_cache_dir",
    "default_samples_dir",
    "fetch_assembly_pdb",
    "fetch_published",
    "fetch_val_split",
    "load_aligned_sample_pdb_bytes",
    "load_assembly_bytes",
    "load_chain_trajectory",
    "load_ensemble_pdb_bytes",
    "load_reference_observables",
    "load_sample_observables",
    "load_sample_pdb_bytes",
    "load_test_chains",
    "load_topology_bytes",
    "load_val_chains",
    "sample_observables_path",
    "sample_path",
]
