from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, cast

import mdtraj as md
import numpy as np
import pytest
import requests

from premval.data import (
    ATLAS_KINDS,
    ATLAS_REPLICAS,
    default_cache_dir,
    fetch_val_split,
    load_chain_trajectory,
    load_val_chains,
)
from premval.data import atlas as atlas_mod


def test_load_val_chains_has_39_entries() -> None:
    chains = load_val_chains()
    assert len(chains) == 39


def test_load_val_chains_names_are_pdb_chain_format() -> None:
    for chain in load_val_chains():
        pdb, _, code = chain.partition("_")
        assert len(pdb) == 4 and pdb.isalnum()
        assert len(code) == 1 and code.isalpha()


def test_atlas_kinds_constant() -> None:
    assert ATLAS_KINDS == ("analysis", "protein", "total")


def test_default_cache_dir_under_home() -> None:
    assert default_cache_dir() == Path.home() / ".cache" / "premval" / "atlas"


class _FakeResponse:
    """Minimal stand-in for `requests.Response` covering the streaming path."""

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
    def __init__(self, payload: bytes = b"FAKE-ATLAS-BUNDLE") -> None:
        self.payload = payload
        self.calls: list[str] = []

    def get(self, url: str, **_: Any) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self.payload)

    def close(self) -> None:
        return None


def test_fetch_val_split_writes_and_caches(tmp_path: Path) -> None:
    session = _FakeSession()
    chains = ["6cka_B", "6hj6_A"]

    first = fetch_val_split(tmp_path, chains=chains, session=cast(requests.Session, session))
    assert set(first) == set(chains)
    for path in first.values():
        assert path.exists() and path.read_bytes() == session.payload
    assert len(session.calls) == len(chains)

    # Second call should hit the cache and not issue any HTTP request.
    second = fetch_val_split(tmp_path, chains=chains, session=cast(requests.Session, session))
    assert second == first
    assert len(session.calls) == len(chains), "cached bundles should not be re-fetched"


def test_fetch_val_split_force_redownloads(tmp_path: Path) -> None:
    session = _FakeSession()
    chains = ["6cka_B"]
    sess = cast(requests.Session, session)
    fetch_val_split(tmp_path, chains=chains, session=sess)
    fetch_val_split(tmp_path, chains=chains, session=sess, force=True)
    assert len(session.calls) == 2


def test_fetch_val_split_propagates_network_error(tmp_path: Path) -> None:
    class _FailingSession:
        def get(self, url: str, **_: Any) -> _FakeResponse:
            raise OSError("network down")

        def close(self) -> None:
            return None

    with pytest.raises(OSError, match="network down"):
        fetch_val_split(
            tmp_path,
            chains=["6cka_B"],
            session=cast(requests.Session, _FailingSession()),
        )


class _FlakySession:
    """Fails with `requests.ConnectionError` for the first `fail_count` calls."""

    def __init__(self, fail_count: int, payload: bytes = b"FAKE-ATLAS-BUNDLE") -> None:
        self.payload = payload
        self.fail_count = fail_count
        self.calls = 0

    def get(self, url: str, **_: Any) -> _FakeResponse:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise requests.ConnectionError("connection reset")
        return _FakeResponse(self.payload)

    def close(self) -> None:
        return None


def test_fetch_val_split_recovers_after_transient_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("premval.data.atlas.time.sleep", lambda _seconds: None)
    session = _FlakySession(fail_count=2)
    result = fetch_val_split(
        tmp_path,
        chains=["6cka_B"],
        session=cast(requests.Session, session),
    )
    assert session.calls == 3
    bundle = result["6cka_B"]
    assert bundle.read_bytes() == session.payload
    assert not bundle.with_suffix(bundle.suffix + ".part").exists()


def test_fetch_val_split_gives_up_after_max_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("premval.data.atlas.time.sleep", lambda _seconds: None)
    session = _FlakySession(fail_count=atlas_mod._BUNDLE_RETRIES)
    with pytest.raises(requests.ConnectionError, match="connection reset"):
        fetch_val_split(
            tmp_path,
            chains=["6cka_B"],
            session=cast(requests.Session, session),
        )
    assert session.calls == atlas_mod._BUNDLE_RETRIES
    leftover = tmp_path / "analysis" / "6cka_B.zip.part"
    assert not leftover.exists(), "partial download should be cleaned up on full failure"


def test_fetch_val_split_closes_owned_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _FakeSession()
    closed = {"count": 0}
    original_close = session.close

    def _tracking_close() -> None:
        closed["count"] += 1
        original_close()

    session.close = _tracking_close  # type: ignore[method-assign]
    monkeypatch.setattr(atlas_mod, "_build_session", lambda: session)

    fetch_val_split(tmp_path, chains=["6cka_B"])
    assert closed["count"] == 1


def test_fetch_val_split_does_not_close_injected_session(tmp_path: Path) -> None:
    session = _FakeSession()
    closed = {"count": 0}
    original_close = session.close

    def _tracking_close() -> None:
        closed["count"] += 1
        original_close()

    session.close = _tracking_close  # type: ignore[method-assign]
    fetch_val_split(tmp_path, chains=["6cka_B"], session=cast(requests.Session, session))
    assert closed["count"] == 0


def _toy_chain_traj(n_frames: int, n_residues: int = 5, seed: int = 0) -> md.Trajectory:
    """Build a tiny CA-only trajectory for synthesizing test bundles."""
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("CA", md.element.carbon, res)
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_frames, n_residues, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def make_full_atom_trajectory(
    n_frames: int, n_residues: int, *, seed: int = 0
) -> md.Trajectory:
    """Build a tiny full-atom (N, CA, C, O) trajectory.

    Used by tests that exercise CA selection: every residue gets four
    backbone atoms so the CA-selection path actually filters, instead of
    short-circuiting on an already-CA-only topology.
    """
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)
    rng = np.random.default_rng(seed)
    n_atoms = n_residues * 4
    xyz = rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def _write_fake_bundle(
    cache_dir: Path,
    chain: str,
    *,
    kind: atlas_mod.AtlasKind = "analysis",
    frames_per_replica: tuple[int, int, int] = (3, 4, 5),
) -> Path:
    """Synthesize a `{chain}.zip` with `{chain}.pdb` + 3 replica XTCs.

    Each replica gets a distinct frame count so the test can verify
    join order (R1+R2+R3) rather than relying on a sum that hides it.
    """
    bundle_dir = cache_dir / kind
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle_dir / f"{chain}.zip"

    topology_traj = _toy_chain_traj(n_frames=1, seed=0)
    staging = cache_dir / "_staging"
    staging.mkdir(exist_ok=True)
    pdb_path = staging / f"{chain}.pdb"
    topology_traj.save_pdb(str(pdb_path))

    xtc_paths: list[Path] = []
    for replica_idx, n_frames in zip(ATLAS_REPLICAS, frames_per_replica, strict=True):
        replica = _toy_chain_traj(n_frames=n_frames, seed=replica_idx)
        xtc_path = staging / f"{chain}_R{replica_idx}.xtc"
        replica.save_xtc(str(xtc_path))
        xtc_paths.append(xtc_path)

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as zf:
        zf.write(pdb_path, arcname=pdb_path.name)
        for xtc_path in xtc_paths:
            zf.write(xtc_path, arcname=xtc_path.name)
    return bundle


def test_load_chain_trajectory_concatenates_three_replicas(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, "fake_A", frames_per_replica=(3, 4, 5))
    traj = load_chain_trajectory("fake_A", cache_dir=tmp_path)
    assert traj.n_frames == 3 + 4 + 5
    assert traj.topology.n_atoms == 5


def test_load_chain_trajectory_missing_bundle_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_chain_trajectory("nope_A", cache_dir=tmp_path)


def test_load_chain_trajectory_missing_xtc_raises(tmp_path: Path) -> None:
    bundle = _write_fake_bundle(tmp_path, "fake_B")
    # Rebuild the zip with R3 missing.
    with zipfile.ZipFile(bundle) as zf:
        members = {name: zf.read(name) for name in zf.namelist() if not name.endswith("_R3.xtc")}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    with pytest.raises(KeyError, match="fake_B_R3.xtc"):
        load_chain_trajectory("fake_B", cache_dir=tmp_path)


def test_build_session_mounts_https_adapter() -> None:
    from requests.adapters import HTTPAdapter

    sess = atlas_mod._build_session()
    try:
        adapter = sess.get_adapter("https://www.dsimb.inserm.fr/")
        assert isinstance(adapter, HTTPAdapter)
        assert adapter.max_retries.total == atlas_mod._SESSION_RETRIES
    finally:
        sess.close()
