"""Ingest published model ensembles into the samples cache.

Several open-weight generators publish their ATLAS ensembles directly, so
the leaderboard can score them for $0 with no GPU (PLAN.md's seeding
strategy). This module turns that ingest into a reproducible command: it
downloads each published archive, extracts the per-target coordinate
files, converts them to the canonical multi-model PDB if needed, and
writes them into the samples cache layout that `samples.py` reads:

    {samples_dir}/{model}/{chain}.pdb

`PUBLISHED_SOURCES` is the registry of what we know how to ingest:

- AlphaFlow / ESMFlow: HuggingFace zips of multi-model PDBs (one per
  target). Extracted and renamed; no coordinate conversion. Only the
  sequence-conditioned ATLAS-MD variants are registered; the `_pdb*`
  (CAMEO predictions, no ATLAS chains) and `*_md_templates*`
  (structure-conditioned) zips are intentionally excluded (see
  `_HF_ATLAS_ZIPS`).
- EBA: a GCS tarball whose targets are multi-model **mmCIF**; converted to
  PDB via mdtraj on the way into the cache.

Member-to-chain mapping is by filename stem matched (case-insensitively)
against the known ATLAS val+test chains, so unexpected archive members are
logged and skipped rather than guessed at.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import requests

from premval.data.atlas import load_test_chains, load_val_chains
from premval.data.samples import default_samples_dir, sample_path

if TYPE_CHECKING:
    from collections.abc import Iterable

_LOG = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_S = 600
_CHUNK_SIZE = 1 << 16

ArchiveFormat = Literal["zip", "tar.gz"]
CoordFormat = Literal["pdb", "cif"]


@dataclass(frozen=True)
class PublishedSource:
    """A downloadable, pre-generated ensemble set for one model.

    Attributes:
        model: Samples-cache key, e.g. `alphaflow_md_base`. Becomes the
            `{samples_dir}/{model}/` directory.
        url: Direct download URL of the archive.
        archive: Container format of the download.
        coord: Coordinate format of the per-target members inside the
            archive. `cif` members are converted to PDB on ingest.
    """

    model: str
    url: str
    archive: ArchiveFormat
    coord: CoordFormat


_HF_SAMPLES = "https://huggingface.co/bjing-mit/alphaflow/resolve/main/samples"

# AlphaFlow / ESMFlow publish per-model ATLAS *test*-split ensembles (82 chains,
# 250 models each) as HuggingFace zips of multi-model PDBs. Only the
# sequence-conditioned ATLAS-MD variants are registered. Two families are
# deliberately excluded:
#   - `_pdb*` zips hold each model's CAMEO/PDB-validation predictions instead
#     (UniProt-named members, zero ATLAS chains): ingesting them downloads
#     hundreds of MB and writes nothing. To put a PDB-trained checkpoint on the
#     ATLAS leaderboard, run inference on the ATLAS split instead
#     (inference/{alphaflow,esmflow}_run.py --checkpoint pdb).
#   - `*_md_templates*` (and the 12-layer `12l` ones) are *structure*-conditioned:
#     they take the reference PDB as a template input, so they're not comparable
#     to the sequence-only generators on this leaderboard and are left out.
# Keys drop the upstream date suffix.
_HF_ATLAS_ZIPS = {
    "alphaflow_md_base": "alphaflow_md_base_202402.zip",
    "alphaflow_md_distilled": "alphaflow_md_distilled_202402.zip",
    "esmflow_md_base": "esmflow_md_base_202402.zip",
    "esmflow_md_distilled": "esmflow_md_distilled_202402.zip",
}

PUBLISHED_SOURCES: dict[str, PublishedSource] = {
    model: PublishedSource(model, f"{_HF_SAMPLES}/{filename}", "zip", "pdb")
    for model, filename in _HF_ATLAS_ZIPS.items()
}
# EBA ships a GCS tarball of multi-model mmCIF, converted to PDB on ingest.
PUBLISHED_SOURCES["eba"] = PublishedSource(
    "eba", "https://storage.googleapis.com/project_icml25_eba/artifacts.tar.gz", "tar.gz", "cif"
)


def _known_chains() -> set[str]:
    """Return the union of ATLAS val and test chain ids (the valid targets)."""
    return set(load_val_chains()) | set(load_test_chains())


def _archive_path(source: PublishedSource, samples_dir: Path) -> Path:
    """Path of the downloaded archive in the `_zips` scratch dir."""
    ext = "zip" if source.archive == "zip" else "tar.gz"
    return samples_dir / "_zips" / f"{source.model}.{ext}"


def _download_archive(
    source: PublishedSource,
    samples_dir: Path,
    *,
    force: bool,
    session: requests.Session | None,
) -> Path:
    """Stream the source archive into `_zips`, returning its path.

    Resumable at the archive level: an already-downloaded archive is reused
    unless `force` is set. Writes via a `.part` temp file so an interrupted
    download never leaves a truncated archive in place.
    """
    dest = _archive_path(source, samples_dir)
    if dest.exists() and not force:
        _LOG.info("%s: archive cached at %s", source.model, dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    sess = session or requests.Session()
    try:
        with sess.get(source.url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
        if session is None:
            sess.close()
    return dest


def _iter_members(source: PublishedSource, archive: Path) -> Iterable[tuple[str, bytes]]:
    """Yield `(member_name, raw_bytes)` for each coordinate member in the archive."""
    suffix = f".{source.coord}"
    if source.archive == "zip":
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.lower().endswith(suffix):
                    yield name, zf.read(name)
        return
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.lower().endswith(suffix):
                handle = tf.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read()


def _write_target(
    raw: bytes,
    coord: CoordFormat,
    dest: Path,
) -> None:
    """Write one target's coordinates to `dest` as a multi-model PDB.

    PDB members are written verbatim; CIF members are round-tripped through
    mdtraj (`load` then `save_pdb`) so the cache is uniformly PDB.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if coord == "pdb":
        dest.write_bytes(raw)
        return
    import mdtraj as md

    with tempfile.TemporaryDirectory() as tmp:
        cif = Path(tmp) / "in.cif"
        cif.write_bytes(raw)
        traj = md.load(str(cif))
        # mdtraj's mmCIF reader leaves atom.serial as a string, which breaks
        # save_pdb's integer formatting; clearing it makes save_pdb renumber.
        for atom in traj.topology.atoms:
            atom.serial = None
        traj.save_pdb(str(dest))


def _extract_archive(
    source: PublishedSource,
    archive: Path,
    samples_dir: Path,
    *,
    chains: Iterable[str] | None,
    force: bool,
) -> dict[str, Path]:
    """Normalize an archive's targets into `{samples_dir}/{model}/{chain}.pdb`.

    Pure filesystem work (no network), so it is unit-testable against a
    synthetic archive. Members whose stem is not a known ATLAS chain are
    logged and skipped; members already cached are skipped unless `force`.

    Args:
        source: The registry entry being ingested.
        archive: Path to the downloaded archive.
        samples_dir: Samples cache root.
        chains: If given, restrict ingest to these chain ids.
        force: Overwrite cached per-target PDBs.

    Returns:
        Mapping of chain id to the written PDB path.
    """
    # Case-insensitive lookup: archive members can carry an upper-case PDB
    # code (common in mmCIF) while the canonical chain id is lower-case.
    canonical = {c.lower(): c for c in _known_chains()}
    wanted = set(chains) if chains is not None else None
    written: dict[str, Path] = {}
    for name, raw in _iter_members(source, archive):
        chain = canonical.get(Path(name).stem.lower())
        if chain is None:
            _LOG.warning("%s: skipping member %r (not a known ATLAS chain)", source.model, name)
            continue
        if wanted is not None and chain not in wanted:
            continue
        dest = sample_path(source.model, chain, samples_dir)
        if dest.exists() and not force:
            _LOG.info("%s/%s: cached", source.model, chain)
            written[chain] = dest
            continue
        _write_target(raw, source.coord, dest)
        written[chain] = dest
    return written


def fetch_published(
    model: str,
    *,
    samples_dir: Path | None = None,
    chains: Iterable[str] | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    """Download and normalize a published ensemble set into the samples cache.

    Args:
        model: A key in `PUBLISHED_SOURCES` (e.g. `alphaflow_md_base`, `eba`).
        samples_dir: Samples cache root. Defaults to `default_samples_dir()`.
        chains: If given, restrict ingest to these chain ids.
        force: Re-download the archive and overwrite cached per-target PDBs.
        session: Optional pre-configured `requests.Session`; a fresh one is
            created and closed if not supplied.

    Returns:
        Mapping of chain id to the written `{model}/{chain}.pdb` path.

    Raises:
        KeyError: If `model` is not in `PUBLISHED_SOURCES`.
        requests.HTTPError: If the archive download returns a non-2xx status.
    """
    if model not in PUBLISHED_SOURCES:
        raise KeyError(f"unknown published model {model!r}; known: {sorted(PUBLISHED_SOURCES)}")
    source = PUBLISHED_SOURCES[model]
    root = samples_dir or default_samples_dir()
    archive = _download_archive(source, root, force=force, session=session)
    return _extract_archive(source, archive, root, chains=chains, force=force)
