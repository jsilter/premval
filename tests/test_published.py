"""Tests for premval.data.published (published-ensemble ingest).

Network is mocked with a fake session (the `test_data` pattern); the
extract/normalize path is exercised against synthetic on-disk archives so
it needs no network at all.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

import mdtraj as md
import numpy as np
import pytest
import requests

from premval.data import sample_path
from premval.data.published import (
    PublishedSource,
    _extract_archive,
    fetch_published,
)

# Real split chains so the known-chain filter does not reject the fixtures.
VAL_CHAIN = "6e33_A"
TEST_CHAIN = "6o2v_A"


def _toy_traj(n_frames: int = 3, n_res: int = 4, seed: int = 0) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_res):
        res = top.add_residue("ALA", chain)
        top.add_atom("CA", md.element.carbon, res)
    xyz = np.random.default_rng(seed).normal(size=(n_frames, n_res, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def _pdb_bytes(n_frames: int = 3) -> bytes:
    traj = _toy_traj(n_frames)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.pdb"
        traj.save_pdb(str(p))
        return p.read_bytes()


def _cif_bytes(n_frames: int = 3) -> bytes:
    traj = _toy_traj(n_frames)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.cif"
        traj.save(str(p))
        return p.read_bytes()


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _make_targz(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _zip_source(model: str = "testmodel") -> PublishedSource:
    return PublishedSource(model, "https://example/x.zip", "zip", "pdb")


def test_extract_zip_pdb_writes_known_chains(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _make_zip(archive, {f"{TEST_CHAIN}.pdb": _pdb_bytes(3), f"{VAL_CHAIN}.pdb": _pdb_bytes(5)})
    written = _extract_archive(_zip_source(), archive, tmp_path, chains=None, force=False)

    assert set(written) == {TEST_CHAIN, VAL_CHAIN}
    for chain, path in written.items():
        assert path == sample_path("testmodel", chain, tmp_path)
        assert path.exists()
    # Coordinates survive the round-trip: frame count preserved.
    assert md.load(str(written[TEST_CHAIN])).n_frames == 3
    assert md.load(str(written[VAL_CHAIN])).n_frames == 5


def test_extract_skips_unknown_members(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _make_zip(archive, {"not_a_chain.pdb": _pdb_bytes(), f"{TEST_CHAIN}.pdb": _pdb_bytes()})
    written = _extract_archive(_zip_source(), archive, tmp_path, chains=None, force=False)
    assert set(written) == {TEST_CHAIN}


def test_extract_chain_filter(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _make_zip(archive, {f"{TEST_CHAIN}.pdb": _pdb_bytes(), f"{VAL_CHAIN}.pdb": _pdb_bytes()})
    written = _extract_archive(_zip_source(), archive, tmp_path, chains=[TEST_CHAIN], force=False)
    assert set(written) == {TEST_CHAIN}


def test_extract_cache_skip_then_force(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _make_zip(archive, {f"{TEST_CHAIN}.pdb": _pdb_bytes(3)})
    dest = sample_path("testmodel", TEST_CHAIN, tmp_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"SENTINEL")

    _extract_archive(_zip_source(), archive, tmp_path, chains=None, force=False)
    assert dest.read_bytes() == b"SENTINEL", "cached target must not be overwritten without force"

    _extract_archive(_zip_source(), archive, tmp_path, chains=None, force=True)
    assert dest.read_bytes() != b"SENTINEL"
    assert md.load(str(dest)).n_frames == 3


def test_extract_cif_converts_to_pdb(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    _make_targz(archive, {f"{TEST_CHAIN}.cif": _cif_bytes(4)})
    source = PublishedSource("eba", "https://example/x.tar.gz", "tar.gz", "cif")
    written = _extract_archive(source, archive, tmp_path, chains=None, force=False)

    assert set(written) == {TEST_CHAIN}
    dest = written[TEST_CHAIN]
    assert dest.suffix == ".pdb"
    assert md.load(str(dest)).n_frames == 4


def test_extract_normalizes_uppercase_member_stem(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    _make_targz(archive, {f"{TEST_CHAIN.upper()}.cif": _cif_bytes(2)})
    source = PublishedSource("eba", "https://example/x.tar.gz", "tar.gz", "cif")
    written = _extract_archive(source, archive, tmp_path, chains=None, force=False)
    assert set(written) == {TEST_CHAIN}


def test_fetch_published_unknown_model_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown published model"):
        fetch_published("nope", samples_dir=tmp_path)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self._payload[i : i + chunk_size] for i in range(0, len(self._payload), chunk_size)]


class _FakeSession:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = 0

    def get(self, url: str, **_: Any) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(self.payload)

    def close(self) -> None:
        return None


def test_fetch_published_downloads_then_reuses_archive(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{TEST_CHAIN}.pdb", _pdb_bytes(3))
    session = _FakeSession(buf.getvalue())

    written = fetch_published(
        "alphaflow_md_base",
        samples_dir=tmp_path,
        session=cast(requests.Session, session),
    )
    assert set(written) == {TEST_CHAIN}
    assert (tmp_path / "_zips" / "alphaflow_md_base.zip").exists()
    assert session.calls == 1

    # Second call: archive is cached, so no new download is issued.
    fetch_published(
        "alphaflow_md_base",
        samples_dir=tmp_path,
        session=cast(requests.Session, session),
    )
    assert session.calls == 1
