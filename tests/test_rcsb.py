"""Tests for premval.data.rcsb metadata fetching.

Network is mocked with a fake session (the `test_data` pattern); no test
hits data.rcsb.org.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import requests

from premval.data.rcsb import (
    EntryMetadata,
    _extract_metadata,
    entry_metadata_cache_path,
    fetch_entry_metadata,
)

# Trimmed copy of a real data.rcsb.org/rest/v1/core/entry/6o2v response.
_RAW_6O2V: dict[str, Any] = {
    "struct": {"title": "Crystal structure of the SARAF luminal domain"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "rcsb_entry_info": {
        "experimental_method": "X-ray",
        "resolution_combined": [1.58],
        "molecular_weight": 31.18,
    },
    "rcsb_accession_info": {
        "deposit_date": "2019-02-24T00:00:00.000+00:00",
        "initial_release_date": "2019-05-29T00:00:00.000+00:00",
    },
    "struct_keywords": {"pdbx_keywords": "SIGNALING PROTEIN"},
    "rcsb_primary_citation": {
        "title": "SARAF Luminal Domain Structure.",
        "rcsb_journal_abbrev": "J Mol Biology",
        "year": 2019,
        "pdbx_database_id_DOI": "10.1016/j.jmb.2019.05.008",
        "rcsb_authors": ["Kimberlin, C.R.", "Minor Jr., D.L."],
    },
}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def get(self, url: str, **_: Any) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(self.payload)


def _as_session(fake: _FakeSession) -> requests.Session:
    return cast(requests.Session, fake)


class TestExtractMetadata:
    def test_picks_display_fields(self) -> None:
        meta = _extract_metadata("6o2v", _RAW_6O2V)
        assert meta.pdb_id == "6o2v"
        assert meta.url == "https://www.rcsb.org/structure/6o2v"
        assert meta.title == "Crystal structure of the SARAF luminal domain"
        assert meta.method == "X-ray"
        assert meta.resolution_a == 1.58
        assert meta.molecular_weight_kda == 31.18
        assert meta.deposit_date == "2019-02-24"
        assert meta.release_date == "2019-05-29"
        assert meta.keywords == "SIGNALING PROTEIN"
        assert meta.citation_doi == "10.1016/j.jmb.2019.05.008"
        assert meta.citation_authors == ["Kimberlin, C.R.", "Minor Jr., D.L."]

    def test_tolerates_missing_fields(self) -> None:
        # An NMR-style sparse record: no resolution, no citation, no keywords.
        meta = _extract_metadata("1abc", {"struct": {"title": "X"}})
        assert meta.title == "X"
        assert meta.resolution_a is None
        assert meta.deposit_date is None
        assert meta.citation_title is None
        assert meta.citation_authors is None


class TestFetchEntryMetadata:
    def test_fetches_then_caches(self, tmp_path: Path) -> None:
        fake = _FakeSession(_RAW_6O2V)
        meta = fetch_entry_metadata("6o2v", cache_dir=tmp_path, session=_as_session(fake))
        assert meta.title == "Crystal structure of the SARAF luminal domain"
        assert entry_metadata_cache_path("6o2v", tmp_path).exists()
        assert fake.calls == 1

        # Second call reads the cache; no new network request.
        cached = fetch_entry_metadata("6o2v", cache_dir=tmp_path, session=_as_session(fake))
        assert cached == meta
        assert fake.calls == 1

    def test_force_refetches(self, tmp_path: Path) -> None:
        fake = _FakeSession(_RAW_6O2V)
        fetch_entry_metadata("6o2v", cache_dir=tmp_path, session=_as_session(fake))
        fetch_entry_metadata("6o2v", cache_dir=tmp_path, session=_as_session(fake), force=True)
        assert fake.calls == 2

    def test_round_trips_through_cache(self, tmp_path: Path) -> None:
        original = fetch_entry_metadata(
            "6o2v", cache_dir=tmp_path, session=_as_session(_FakeSession(_RAW_6O2V))
        )
        reloaded = fetch_entry_metadata("6o2v", cache_dir=tmp_path)
        assert isinstance(reloaded, EntryMetadata)
        assert reloaded == original

    def test_rejects_malformed_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid PDB id"):
            fetch_entry_metadata(
                "nope!", cache_dir=tmp_path, session=_as_session(_FakeSession(_RAW_6O2V))
            )
