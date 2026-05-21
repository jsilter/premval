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

import numpy as np
from numpy.typing import NDArray

from premval.data.references import (
    ReferenceObservables,
    compute_observables_from_traj,
    kabsch_matrix,
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


def aligned_sample_path(model: str, chain: str, samples_dir: Path | None = None) -> Path:
    """Cache path for the rigid-body-aligned sample PDB.

    Stored in an underscore-prefixed dir so it isn't mistaken for a model:
    `{dir}/_aligned/{model}/{chain}.pdb`.
    """
    root = samples_dir or default_samples_dir()
    return root / "_aligned" / model / f"{chain}.pdb"


def _build_aligned_sample(
    model: str,
    chain: str,
    reference_ca_nm: NDArray[np.floating],
    samples_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Chain-align a sample ensemble to remove rigid-body tumbling; cache it.

    Generative samples (notably AlphaFlow) emit each conformation in an
    arbitrary global orientation, so raw playback tumbles wildly and obscures
    the actual conformational change. We remove that overall component with a
    two-step rigid alignment (CA atoms, no scaling):

    1. Frame 0 is superposed onto `reference_ca_nm` (the reference's frame-0
       CA), anchoring the sample to the reference's orientation.
    2. Every later frame is superposed onto the *previous* (already-aligned)
       frame, so consecutive frames differ minimally and playback is smooth.

    The result is cached as a multi-model PDB and reused. Internal motion is
    preserved; only the overall rotation/translation per frame is changed.

    Args:
        model: Sample model key.
        chain: PDB chain identifier.
        reference_ca_nm: Reference frame-0 CA coordinates, shape `(n_res, 3)`,
            in nanometers (same residue order as the sample's CA atoms).
        samples_dir: Samples cache root.
        force: Rebuild even if the aligned PDB is already cached.

    Returns:
        Path to the cached aligned multi-model PDB.

    Raises:
        FileNotFoundError: If no raw sample PDB exists for `(model, chain)`.
    """
    cache = aligned_sample_path(model, chain, samples_dir)
    if cache.exists() and not force:
        return cache
    raw = sample_path(model, chain, samples_dir)
    if not raw.exists():
        raise FileNotFoundError(
            f"no cached sample at {raw}; expected {model!r} ensemble for chain {chain!r}"
        )

    import mdtraj as md

    traj = md.load(str(raw))
    ca = traj.topology.select("name CA")
    xyz = traj.xyz  # (n_frames, n_atoms, 3), nm, float32
    ref = np.asarray(reference_ca_nm, dtype=xyz.dtype)

    def apply(frame: NDArray[np.float32], target: NDArray[np.floating]) -> NDArray[np.float32]:
        m = kabsch_matrix(frame[ca], target)
        rot = m[:3, :3].T.astype(xyz.dtype)
        trans = m[:3, 3].astype(xyz.dtype)
        moved: NDArray[np.float32] = frame @ rot + trans
        return moved

    xyz[0] = apply(xyz[0], ref)
    for n in range(1, xyz.shape[0]):
        xyz[n] = apply(xyz[n], xyz[n - 1][ca])
    traj.xyz = xyz

    cache.parent.mkdir(parents=True, exist_ok=True)
    traj.save_pdb(str(cache))
    return cache


def load_aligned_sample_pdb_bytes(
    model: str,
    chain: str,
    reference_ca_nm: NDArray[np.floating],
    samples_dir: Path | None = None,
    *,
    force: bool = False,
) -> bytes:
    """Return the rigid-body-aligned sample PDB bytes (built + cached on first call).

    See `_build_aligned_sample` for the alignment scheme. This is what the
    viewer streams so the model is oriented like the reference and does not
    tumble during playback.
    """
    return _build_aligned_sample(
        model, chain, reference_ca_nm, samples_dir, force=force
    ).read_bytes()


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
    reference_ca_nm: NDArray[np.floating],
    samples_dir: Path | None = None,
    *,
    force: bool = False,
) -> ReferenceObservables:
    """Compute (and cache) the observables panel for a model's sample ensemble.

    Computed from the **rigid-body-aligned** sample (see
    `_build_aligned_sample`), so the overlay frame matches the coordinates the
    viewer displays — PCA-mode sweeps and ellipsoid axes line up with the
    aligned structure rather than the raw, tumbling one. Uses the same
    `compute_observables_from_traj` pipeline as the ATLAS reference, so the two
    panes' overlays are directly comparable. First call computes + saves;
    later calls memo-load from `sample_observables_path`.

    Args:
        model: Sample model key, e.g. `alphaflow_md_base`.
        chain: PDB chain identifier.
        reference_ca_nm: Reference frame-0 CA coordinates (nm) used to anchor
            the sample alignment.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        force: Recompute and overwrite an existing cache entry.

    Returns:
        The computed `ReferenceObservables` for the aligned sample ensemble.

    Raises:
        FileNotFoundError: If no cached sample PDB exists for `(model, chain)`.
    """
    cache = sample_observables_path(model, chain, samples_dir)
    if cache.exists() and not force:
        return load_observables(cache)

    import mdtraj as md

    aligned = _build_aligned_sample(model, chain, reference_ca_nm, samples_dir, force=force)
    obs = compute_observables_from_traj(md.load(str(aligned)))
    save_observables(obs, cache)
    return obs
