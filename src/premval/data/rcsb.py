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

import logging
import re
from pathlib import Path

import requests

from premval.data.atlas import default_cache_dir

_RCSB_BASE = "https://files.rcsb.org/download"
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
