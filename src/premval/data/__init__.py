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
    load_chain_sequences,
    load_chain_trajectory,
    load_ensemble_pdb_bytes,
    load_test_chains,
    load_topology_bytes,
    load_val_chains,
    load_view_ensemble_pdb_bytes,
    view_ensemble_path,
)
from premval.data.published import PUBLISHED_SOURCES, PublishedSource, fetch_published
from premval.data.rcsb import (
    EntryMetadata,
    assembly_cache_path,
    entry_metadata_cache_path,
    fetch_assembly_pdb,
    fetch_entry_metadata,
    load_assembly_bytes,
)
from premval.data.references import ReferenceObservables, load_reference_observables
from premval.data.samples import (
    aligned_sample_path,
    available_chains,
    available_models,
    default_samples_dir,
    load_aligned_sample_pdb_bytes,
    load_sample_metrics,
    load_sample_observables,
    load_sample_pdb_bytes,
    load_view_sample_pdb_bytes,
    sample_metrics_path,
    sample_observables_path,
    sample_path,
    warm_view_caches,
)

__all__ = [
    "ATLAS_KINDS",
    "ATLAS_REPLICAS",
    "PUBLISHED_SOURCES",
    "AtlasKind",
    "EntryMetadata",
    "PublishedSource",
    "ReferenceObservables",
    "aligned_sample_path",
    "assembly_cache_path",
    "entry_metadata_cache_path",
    "available_chains",
    "available_models",
    "bundle_path",
    "default_cache_dir",
    "default_samples_dir",
    "fetch_assembly_pdb",
    "fetch_entry_metadata",
    "fetch_published",
    "fetch_val_split",
    "load_aligned_sample_pdb_bytes",
    "load_assembly_bytes",
    "load_chain_sequences",
    "load_chain_trajectory",
    "load_ensemble_pdb_bytes",
    "load_reference_observables",
    "load_sample_metrics",
    "load_sample_observables",
    "load_sample_pdb_bytes",
    "load_test_chains",
    "load_topology_bytes",
    "load_val_chains",
    "load_view_ensemble_pdb_bytes",
    "load_view_sample_pdb_bytes",
    "view_ensemble_path",
    "sample_metrics_path",
    "sample_observables_path",
    "sample_path",
    "warm_view_caches",
]
