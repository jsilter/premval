"""Access cached generative-model sample ensembles.

Published AlphaFlow / ESMFlow inference outputs are multi-model PDBs (one
MODEL per sampled conformation, 250 frames per target) covering the ATLAS
test split. They are stored per model under the samples cache:

    {samples_dir}/{model}/{chain}.pdb

where `model` is a key like `alphaflow_md_base` or `esmflow_md_base`. This
module locates and reads them for the side-by-side viewer (and, later,
scoring). Underscore-prefixed directories (e.g. `_zips`) are scratch space
and are not treated as models.
"""

from __future__ import annotations

from pathlib import Path

from premval.data.references import (
    ReferenceObservables,
    compute_observables_from_traj,
    load_observables,
    save_observables,
)


def default_samples_dir() -> Path:
    """Return the default samples cache: `~/.cache/premval/samples/`.

    Sibling of the ATLAS cache (`~/.cache/premval/atlas/`) so all premval
    data lives under one predictable root.
    """
    return Path.home() / ".cache" / "premval" / "samples"


def sample_path(model: str, chain: str, samples_dir: Path | None = None) -> Path:
    """Path to a model's sample ensemble for a chain: `{dir}/{model}/{chain}.pdb`."""
    root = samples_dir or default_samples_dir()
    return root / model / f"{chain}.pdb"


def available_models(samples_dir: Path | None = None) -> list[str]:
    """Return sample model keys present in the cache, sorted.

    Skips underscore-prefixed scratch directories (e.g. `_zips`).
    """
    root = samples_dir or default_samples_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))


def available_chains(model: str, samples_dir: Path | None = None) -> list[str]:
    """Return chain ids that `model` has a cached sample ensemble for, sorted."""
    root = (samples_dir or default_samples_dir()) / model
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.pdb"))


def load_sample_pdb_bytes(model: str, chain: str, samples_dir: Path | None = None) -> bytes:
    """Return the multi-model PDB bytes of `model`'s sample ensemble for `chain`.

    Served as-is: published AlphaFlow/ESMFlow outputs already meet the
    250-frame ensemble contract, so no subsampling is applied.

    Args:
        model: Sample model key, e.g. `alphaflow_md_base`.
        chain: PDB chain identifier, e.g. `6o2v_A`.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.

    Returns:
        Multi-model PDB contents as raw bytes.

    Raises:
        FileNotFoundError: If no cached sample exists for `(model, chain)`.
    """
    path = sample_path(model, chain, samples_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached sample at {path}; expected {model!r} ensemble for chain {chain!r}"
        )
    return path.read_bytes()


def sample_observables_path(model: str, chain: str, samples_dir: Path | None = None) -> Path:
    """Cache path for a sample ensemble's observables `.npz`.

    Kept under the samples cache in an underscore-prefixed dir so it is not
    mistaken for a model by `available_models`:
    `{dir}/_observables/{model}/{chain}.npz`.
    """
    root = samples_dir or default_samples_dir()
    return root / "_observables" / model / f"{chain}.npz"


def load_sample_observables(
    model: str,
    chain: str,
    samples_dir: Path | None = None,
    *,
    force: bool = False,
) -> ReferenceObservables:
    """Compute (and cache) the observables panel for a model's sample ensemble.

    Runs the same computation as the ATLAS reference path
    (`compute_observables_from_traj`) on the sample's multi-model PDB, so the
    sample overlays (flexibility, contacts, ellipsoids, PCA) are directly
    comparable to the reference's. First call computes + saves; later calls
    memo-load from `sample_observables_path`.

    Args:
        model: Sample model key, e.g. `alphaflow_md_base`.
        chain: PDB chain identifier.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        force: Recompute and overwrite an existing cache entry.

    Returns:
        The computed `ReferenceObservables` for the sample ensemble.

    Raises:
        FileNotFoundError: If no cached sample PDB exists for `(model, chain)`.
    """
    pdb = sample_path(model, chain, samples_dir)
    if not pdb.exists():
        raise FileNotFoundError(
            f"no cached sample at {pdb}; expected {model!r} ensemble for chain {chain!r}"
        )
    cache = sample_observables_path(model, chain, samples_dir)
    if cache.exists() and not force:
        return load_observables(cache)

    import mdtraj as md

    obs = compute_observables_from_traj(md.load(str(pdb)))
    save_observables(obs, cache)
    return obs
