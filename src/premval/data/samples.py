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

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from premval.data.atlas import bundle_path, default_cache_dir
from premval.data.references import (
    ReferenceObservables,
    compute_observables_from_traj,
    kabsch_matrix,
    load_observables,
    load_reference_observables,
    save_observables,
)


def default_samples_dir() -> Path:
    """Return the samples cache root.

    Honors the ``PREMVAL_SAMPLES_DIR`` environment variable when set (used to
    redirect the cache onto a mounted volume in containerized runs, e.g. the
    Modal harnesses); otherwise falls back to ``~/.cache/premval/samples/``, a
    sibling of the ATLAS cache (``~/.cache/premval/atlas/``) so all premval data
    lives under one predictable root.
    """
    env = os.environ.get("PREMVAL_SAMPLES_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "premval" / "samples"


def sample_path(model: str, chain: str, samples_dir: Path | None = None) -> Path:
    """Path to a model's sample ensemble for a chain: `{dir}/{model}/{chain}.pdb`."""
    root = samples_dir or default_samples_dir()
    return root / model / f"{chain}.pdb"


def available_models(samples_dir: Path | None = None) -> list[str]:
    """Return sample model keys discoverable in the cache, sorted.

    A model counts as discoverable if it has either a raw ensemble directory
    (`{model}/`) or a precomputed view-sidecar directory (`_view/{model}/`).
    The view fallback lets a read-only deployment list models whose multi-GB
    raw ensembles were never uploaded but whose viewer artifacts (built by
    `warm_view_caches`) were: the viewer streams the `_view`/`_observables`/
    `_metrics` sidecars, never the raw ensemble, so a sidecar is sufficient to
    show the model. Skips other underscore-prefixed scratch directories
    (e.g. `_zips`, `_aligned`).
    """
    root = samples_dir or default_samples_dir()
    if not root.exists():
        return []
    raw = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")}
    view_root = root / "_view"
    viewed = {p.name for p in view_root.iterdir() if p.is_dir()} if view_root.exists() else set()
    return sorted(raw | viewed)


def available_chains(model: str, samples_dir: Path | None = None) -> list[str]:
    """Return chain ids `model` can be shown for, sorted.

    A chain counts if `model` has either a raw ensemble (`{model}/{chain}.pdb`)
    or a precomputed view sidecar (`_view/{model}/{chain}.pdb`), mirroring the
    raw-or-sidecar discovery in `available_models` so a model served entirely
    from sidecars still resolves its chains.
    """
    root = samples_dir or default_samples_dir()
    chains: set[str] = set()
    for source in (root / model, root / "_view" / model):
        if source.exists():
            chains.update(p.stem for p in source.glob("*.pdb"))
    return sorted(chains)


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


def view_sample_path(model: str, chain: str, samples_dir: Path | None = None) -> Path:
    """Cache path for the frame-trimmed sample PDB streamed to the viewer.

    Underscore-prefixed so it isn't mistaken for a model by `available_models`:
    `{dir}/_view/{model}/{chain}.pdb`.
    """
    root = samples_dir or default_samples_dir()
    return root / "_view" / model / f"{chain}.pdb"


def load_view_sample_pdb_bytes(
    model: str,
    chain: str,
    reference_ca_nm: NDArray[np.floating],
    samples_dir: Path | None = None,
    *,
    max_frames: int = 120,
    force: bool = False,
) -> bytes:
    """Return the aligned sample PDB trimmed to `max_frames`, for viewer playback.

    Built from the cached rigid-body-aligned ensemble (see
    `_build_aligned_sample`) and stride-subsampled so the kept frames stay in
    order: the aligned ensemble has each frame superposed on the previous one,
    so an evenly strided subset preserves the frame-to-frame smoothness that
    makes playback readable (a random subset would reintroduce jumps). Frame
    count does not affect atom indexing, so the viewer's `ca_indices`-based
    overlays and the separately-computed metric panel are untouched. First call
    builds + caches; later calls memo-load from `view_sample_path`.

    Args:
        model: Sample model key, e.g. `alphaflow_md_base`.
        chain: PDB chain identifier.
        reference_ca_nm: Reference frame-0 CA coordinates (nm) used to anchor
            the underlying sample alignment.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        max_frames: Cap on the number of MODEL records streamed to the viewer.
        force: Rebuild the aligned source and this view even if cached.

    Returns:
        Multi-model PDB contents as raw bytes (at most `max_frames` models).

    Raises:
        FileNotFoundError: If no cached sample PDB exists for `(model, chain)`.
    """
    cache = view_sample_path(model, chain, samples_dir)
    if cache.exists() and not force:
        return cache.read_bytes()

    import mdtraj as md

    aligned = _build_aligned_sample(model, chain, reference_ca_nm, samples_dir, force=force)
    traj = md.load(str(aligned))
    if traj.n_frames > max_frames:
        idx = np.linspace(0, traj.n_frames - 1, max_frames).round().astype(int)
        traj = traj[idx]
    cache.parent.mkdir(parents=True, exist_ok=True)
    traj.save_pdb(str(cache))
    return cache.read_bytes()


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


def sample_metrics_path(model: str, chain: str, samples_dir: Path | None = None) -> Path:
    """Cache path for a sample ensemble's scored metric panel `.json`.

    Kept under the samples cache in an underscore-prefixed dir so it is not
    mistaken for a model by `available_models`:
    `{dir}/_metrics/{model}/{chain}.json`.
    """
    root = samples_dir or default_samples_dir()
    return root / "_metrics" / model / f"{chain}.json"


def load_sample_metrics(
    model: str,
    chain: str,
    *,
    cache_dir: Path | None = None,
    samples_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compute (and cache) the metric panel for a model's sample vs the reference.

    Runs `premval.scoring.score_chain` on the model's raw sample ensemble
    against the cached ATLAS reference for `chain`, producing the same panel
    (`rmwd`, `rmsf_pearson`, `md_pca_w2`, contact Jaccards, plus components)
    the leaderboard records. First call computes + saves; later calls
    memo-load from `sample_metrics_path`.

    Unlike the leaderboard, no frame-count contract is enforced: the viewer
    scores whatever sample ensemble is cached so any chain it can already show
    side by side also gets metrics.

    Args:
        model: Sample model key, e.g. `alphaflow_md_base`.
        chain: PDB chain identifier.
        cache_dir: ATLAS cache root (for the reference). Defaults to the
            ATLAS default cache.
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        force: Recompute and overwrite an existing cache entry.

    Returns:
        The metric dict from `score_chain` (includes `chain` and
        `submission_path`).

    Raises:
        FileNotFoundError: If no cached sample PDB exists for `(model, chain)`.
    """
    cache = sample_metrics_path(model, chain, samples_dir)
    if cache.exists() and not force:
        result: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8"))
        return result

    sample = sample_path(model, chain, samples_dir)
    if not sample.exists():
        raise FileNotFoundError(
            f"no cached sample at {sample}; expected {model!r} ensemble for chain {chain!r}"
        )

    # Lazy import: scoring imports from premval.data, so a module-level import
    # here would be circular. mdtraj is likewise imported lazily across this
    # module.
    from premval.scoring import score_chain

    result = score_chain(sample, chain, cache_dir=cache_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def warm_view_caches(
    models: Sequence[str] | None = None,
    *,
    cache_dir: Path | None = None,
    samples_dir: Path | None = None,
    kind: str = "analysis",
    overwrite: bool = False,
) -> dict[str, tuple[int, int, int]]:
    """Precompute the per-(model, chain) viewer sidecars for the dashboard.

    For every chain a model has a cached sample for and that has a cached
    reference bundle, builds the four artifacts the chain page would otherwise
    compute on first request: the rigid-body-aligned ensemble, the trimmed view
    PDB, the sample observables `.npz`, and the scored metric `.json`. This lets
    the dashboard run read-only (e.g. served from a Modal volume that does not
    persist per-request writes), so no viewer hit triggers a multi-second
    recompute.

    Idempotent: a pair whose observables and metrics caches both exist is
    skipped unless `overwrite` is set. A pair that fails to build (e.g. an
    unreadable ensemble) is logged and counted, and does not abort the batch.

    Args:
        models: Sample model keys to warm; defaults to every cached model.
        cache_dir: ATLAS cache root holding the reference bundles/observables.
        samples_dir: Samples cache root holding the ensembles and sidecars.
        kind: ATLAS payload tier of the reference bundles.
        overwrite: Rebuild caches that already exist.

    Returns:
        Per model, a `(built, skipped, failed)` count triple.
    """
    cache_root = cache_dir or default_cache_dir()
    sample_root = samples_dir or default_samples_dir()
    chosen = list(models) if models is not None else available_models(sample_root)
    summary: dict[str, tuple[int, int, int]] = {}
    for model in chosen:
        built = skipped = failed = 0
        for chain in available_chains(model, sample_root):
            if not bundle_path(cache_root, kind, chain).exists():
                continue
            cached = (
                sample_observables_path(model, chain, sample_root).exists()
                and sample_metrics_path(model, chain, sample_root).exists()
            )
            if cached and not overwrite:
                skipped += 1
                continue
            try:
                ref_ca = load_reference_observables(
                    chain, kind=kind, cache_dir=cache_root
                ).crystal_xyz_ca
                load_sample_observables(model, chain, ref_ca, sample_root, force=overwrite)
                load_view_sample_pdb_bytes(model, chain, ref_ca, sample_root, force=overwrite)
                load_sample_metrics(
                    model, chain, cache_dir=cache_root, samples_dir=sample_root, force=overwrite
                )
                built += 1
            except Exception:  # noqa: BLE001
                # Batch over many independent ensembles: one unreadable or
                # malformed sample must not abort warming the rest. The failure
                # is logged and counted; the CLI exits nonzero if any failed.
                logging.exception("failed to warm %s/%s", model, chain)
                failed += 1
        summary[model] = (built, skipped, failed)
    return summary
