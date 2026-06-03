"""Fetch and cache full structures from RCSB PDB.

ATLAS chain bundles are per-chain extractions; to view a chain in the
context of its parent assembly (the other chains it crystallized with),
we pull the full structure from RCSB.

The RCSB download endpoint (`https://files.rcsb.org/download/{id}.pdb`)
serves the asymmetric unit in legacy PDB format with no auth. That covers
all 39 val-split structures (the largest, 5yrv, has 12 chains and is
~1.8 MB; well under any need for streaming).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from premval.data.atlas import default_cache_dir

_RCSB_BASE = "https://files.rcsb.org/download"
_RCSB_DATA_BASE = "https://data.rcsb.org/rest/v1/core/entry"
_DOWNLOAD_TIMEOUT_S = 60
_PDB_ID_RE = re.compile(r"^[a-zA-Z0-9]{4}$")

_LOG = logging.getLogger(__name__)


def _validate_pdb_id(pdb_id: str) -> str:
    """Return lowercased `pdb_id` after validating it as a 4-char PDB ID.

    Raises:
        ValueError: If `pdb_id` is not exactly 4 alphanumeric characters.
            Guards against path traversal in `assembly_cache_path`.
    """
    if not _PDB_ID_RE.match(pdb_id):
        raise ValueError(f"invalid PDB id {pdb_id!r}; expected 4 alphanumeric chars")
    return pdb_id.lower()


def assembly_cache_path(pdb_id: str, cache_dir: Path | None = None) -> Path:
    """Return the on-disk path for a cached RCSB PDB file.

    Layout: `{cache_dir}/rcsb/{pdb_id}.pdb` (sibling of the per-kind atlas
    bundle dirs). Keeps the RCSB cache discoverable under the same root
    the user already points `--cache-dir` at.
    """
    root = cache_dir or default_cache_dir()
    return root / "rcsb" / f"{_validate_pdb_id(pdb_id)}.pdb"


def fetch_assembly_pdb(
    pdb_id: str,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Download `{pdb_id}.pdb` from RCSB into the cache if not already present.

    Args:
        pdb_id: 4-character PDB entry ID (case-insensitive).
        cache_dir: Root cache directory; the file is written to
            `{cache_dir}/rcsb/{pdb_id}.pdb`. Defaults to
            `default_cache_dir()`.
        force: Re-download even if the cached file exists.
        session: Optional pre-configured `requests.Session`.

    Returns:
        Path to the on-disk PDB file. The file is guaranteed to exist.

    Raises:
        ValueError: If `pdb_id` is malformed.
        requests.HTTPError: If RCSB returns a non-2xx status (e.g., 404
            for a PDB ID that does not exist).
    """
    out = assembly_cache_path(pdb_id, cache_dir)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    url = f"{_RCSB_BASE}/{_validate_pdb_id(pdb_id)}.pdb"
    _LOG.info("fetching %s from RCSB", pdb_id)
    response = sess.get(url, timeout=_DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    tmp = out.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(out)
    return out


def load_assembly_bytes(
    pdb_id: str,
    *,
    cache_dir: Path | None = None,
) -> bytes:
    """Return the full-assembly PDB bytes for `pdb_id`, fetching if needed.

    Convenience wrapper around `fetch_assembly_pdb` for callers (the web
    app) that only want the raw bytes to stream to the browser.
    """
    return fetch_assembly_pdb(pdb_id, cache_dir=cache_dir).read_bytes()


@dataclass(frozen=True)
class EntryMetadata:
    """Curated subset of an RCSB PDB entry's metadata, for display.

    Every field except `pdb_id` and `url` is optional because the RCSB
    record varies by experimental method and curation completeness (NMR
    structures have no resolution; some entries lack a primary citation).
    """

    pdb_id: str
    url: str
    title: str | None = None
    method: str | None = None
    resolution_a: float | None = None
    deposit_date: str | None = None
    release_date: str | None = None
    keywords: str | None = None
    molecular_weight_kda: float | None = None
    citation_title: str | None = None
    citation_journal: str | None = None
    citation_year: int | None = None
    citation_doi: str | None = None
    citation_authors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the metadata fields."""
        return asdict(self)


def entry_metadata_cache_path(pdb_id: str, cache_dir: Path | None = None) -> Path:
    """Path for the cached curated entry metadata JSON.

    Layout: `{cache_dir}/rcsb/{pdb_id}.metadata.json` (alongside the cached
    assembly PDB so all RCSB-sourced data for an entry sits together).
    """
    root = cache_dir or default_cache_dir()
    return root / "rcsb" / f"{_validate_pdb_id(pdb_id)}.metadata.json"


def _first_date(value: str | None) -> str | None:
    """Return the `YYYY-MM-DD` prefix of an RCSB ISO timestamp, or None."""
    return value.split("T", 1)[0] if value else None


def _extract_metadata(pdb_id: str, raw: dict[str, Any]) -> EntryMetadata:
    """Pick the display fields out of a raw RCSB entry record.

    Pure (no I/O); `raw` is the JSON body of
    `data.rcsb.org/rest/v1/core/entry/{id}`. Tolerant of missing keys so a
    sparsely curated entry still yields a usable record.
    """
    entry_info = raw.get("rcsb_entry_info", {})
    accession = raw.get("rcsb_accession_info", {})
    citation = raw.get("rcsb_primary_citation", {})
    resolution = entry_info.get("resolution_combined")
    return EntryMetadata(
        pdb_id=pdb_id,
        url=f"https://www.rcsb.org/structure/{pdb_id}",
        title=raw.get("struct", {}).get("title"),
        method=entry_info.get("experimental_method"),
        resolution_a=resolution[0] if resolution else None,
        deposit_date=_first_date(accession.get("deposit_date")),
        release_date=_first_date(accession.get("initial_release_date")),
        keywords=raw.get("struct_keywords", {}).get("pdbx_keywords"),
        molecular_weight_kda=entry_info.get("molecular_weight"),
        citation_title=citation.get("title"),
        citation_journal=citation.get("rcsb_journal_abbrev"),
        citation_year=citation.get("year"),
        citation_doi=citation.get("pdbx_database_id_DOI"),
        citation_authors=citation.get("rcsb_authors"),
    )


def fetch_entry_metadata(
    pdb_id: str,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> EntryMetadata:
    """Fetch curated metadata for a PDB entry, caching the result.

    Reads from the RCSB data API (`data.rcsb.org`), distinct from the file
    download endpoint used by `fetch_assembly_pdb`. The curated subset is
    cached as JSON so repeat views (and tests) don't re-hit the network.

    Args:
        pdb_id: 4-character PDB entry ID (case-insensitive).
        cache_dir: Root cache directory; metadata is written to
            `{cache_dir}/rcsb/{pdb_id}.metadata.json`. Defaults to
            `default_cache_dir()`.
        force: Re-fetch even if the cached metadata exists.
        session: Optional pre-configured `requests.Session`.

    Returns:
        The curated `EntryMetadata`.

    Raises:
        ValueError: If `pdb_id` is malformed.
        requests.HTTPError: If RCSB returns a non-2xx status.
    """
    pdb_id = _validate_pdb_id(pdb_id)
    cache = entry_metadata_cache_path(pdb_id, cache_dir)
    if cache.exists() and not force:
        return EntryMetadata(**json.loads(cache.read_text()))
    sess = session or requests.Session()
    _LOG.info("fetching metadata for %s from RCSB", pdb_id)
    response = sess.get(f"{_RCSB_DATA_BASE}/{pdb_id}", timeout=_DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    meta = _extract_metadata(pdb_id, response.json())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(meta.to_dict()))
    return meta
