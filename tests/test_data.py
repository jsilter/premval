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
    available_chains,
    available_models,
    default_cache_dir,
    default_samples_dir,
    fetch_val_split,
    load_chain_sequences,
    load_chain_trajectory,
    load_ensemble_pdb_bytes,
    load_reference_observables,
    load_sample_pdb_bytes,
    load_test_chains,
    load_topology_bytes,
    load_val_chains,
    load_view_ensemble_pdb_bytes,
    sample_metrics_path,
    sample_observables_path,
    sample_path,
    view_ensemble_path,
    warm_view_caches,
)
from premval.data import atlas as atlas_mod
from premval.data import samples as samples_mod


def test_load_val_chains_has_39_entries() -> None:
    chains = load_val_chains()
    assert len(chains) == 39


def test_load_val_chains_names_are_pdb_chain_format() -> None:
    for chain in load_val_chains():
        pdb, _, code = chain.partition("_")
        assert len(pdb) == 4 and pdb.isalnum()
        assert len(code) == 1 and code.isalpha()


def test_load_test_chains_has_82_entries() -> None:
    assert len(load_test_chains()) == 82


def test_test_and_val_splits_are_disjoint() -> None:
    # AlphaFlow's published samples cover only the test split; the leaderboard
    # relies on the two splits not overlapping.
    assert set(load_test_chains()).isdisjoint(load_val_chains())


def test_load_chain_sequences_keys_match_chain_loader() -> None:
    # Same chains as `load_val_chains`, now carrying their seqres (one-letter,
    # non-empty) for MSA generation / sequence-conditioned samplers.
    seqs = load_chain_sequences("val")
    assert list(seqs) == load_val_chains()
    assert all(seq and seq.isalpha() for seq in seqs.values())
    assert len(load_chain_sequences("test")) == 82


def test_load_chain_sequences_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        load_chain_sequences("train")


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


def make_full_atom_trajectory(n_frames: int, n_residues: int, *, seed: int = 0) -> md.Trajectory:
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
    kind: str = "analysis",
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


def test_non_atlas_namespace_bundle_loads_and_scores(tmp_path: Path) -> None:
    """A bundle in a non-ATLAS namespace is a drop-in for the ATLAS loaders.

    This is the linchpin of the nanobody-MD plan: any pipeline that emits the
    ATLAS bundle layout (`{id}.pdb` + `{id}_R{1,2,3}.xtc`) under its own cache
    namespace can reuse `load_chain_trajectory` and `load_reference_observables`
    unchanged. Proving it for a `nanobody` namespace guards that contract.
    """
    _write_fake_bundle(tmp_path, "nb_A", kind="nanobody", frames_per_replica=(3, 4, 5))

    traj = load_chain_trajectory("nb_A", kind="nanobody", cache_dir=tmp_path)
    assert traj.n_frames == 3 + 4 + 5

    obs = load_reference_observables("nb_A", "nanobody", cache_dir=tmp_path)
    assert obs.ca_indices.shape[0] == traj.topology.n_atoms
    assert (tmp_path / "references" / "nanobody" / "nb_A.npz").exists()


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


def test_load_topology_bytes_returns_pdb_member(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, "fake_C")
    raw = load_topology_bytes("fake_C", cache_dir=tmp_path)
    # mdtraj-saved PDBs start with REMARK/CRYST1/HEADER (or ATOM if the
    # topology had no metadata, as our synthetic bundles do).
    head = raw[:80].decode("ascii")
    assert head.startswith(("REMARK", "CRYST1", "MODEL", "HEADER", "ATOM"))


def test_load_topology_bytes_missing_bundle_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_topology_bytes("nope_A", cache_dir=tmp_path)


def test_load_topology_bytes_missing_member_raises(tmp_path: Path) -> None:
    bundle = _write_fake_bundle(tmp_path, "fake_D")
    # Drop the topology PDB, keep the XTCs.
    with zipfile.ZipFile(bundle) as zf:
        members = {name: zf.read(name) for name in zf.namelist() if not name.endswith(".pdb")}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    with pytest.raises(KeyError, match="fake_D.pdb"):
        load_topology_bytes("fake_D", cache_dir=tmp_path)


def test_load_ensemble_pdb_bytes_caps_frames(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, "fake_E", frames_per_replica=(20, 20, 20))  # 60 frames total
    raw = load_ensemble_pdb_bytes("fake_E", cache_dir=tmp_path, max_frames=10)
    text = raw.decode("ascii")
    # mdtraj writes one "MODEL ..." line per frame in multi-model PDB.
    assert text.count("\nMODEL ") + text.startswith("MODEL ") == 10


def test_load_ensemble_pdb_bytes_passes_through_small_traj(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, "fake_F", frames_per_replica=(2, 3, 4))  # 9 frames total
    raw = load_ensemble_pdb_bytes("fake_F", cache_dir=tmp_path, max_frames=250)
    text = raw.decode("ascii")
    assert text.count("\nMODEL ") + text.startswith("MODEL ") == 9


def test_load_view_ensemble_pdb_bytes_builds_caches_and_caps(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, "fake_G", frames_per_replica=(20, 20, 20))  # 60 frames
    out = load_view_ensemble_pdb_bytes("fake_G", cache_dir=tmp_path, max_frames=10)
    # Built file matches a direct (uncached) ensemble load at the same cap, and
    # is capped to max_frames MODEL records.
    assert out == load_ensemble_pdb_bytes("fake_G", cache_dir=tmp_path, max_frames=10)
    assert out.decode("ascii").count("\nMODEL ") + out.startswith(b"MODEL ") == 10
    assert view_ensemble_path("fake_G", "analysis", tmp_path).exists()


def test_load_view_ensemble_pdb_bytes_reads_cache_without_bundle(tmp_path: Path) -> None:
    # First call builds + caches the view PDB; once cached, the source bundle is
    # no longer needed (the read-only deployment serves the cache alone).
    _write_fake_bundle(tmp_path, "fake_H", frames_per_replica=(5, 5, 5))
    first = load_view_ensemble_pdb_bytes("fake_H", cache_dir=tmp_path, max_frames=10)
    (tmp_path / "analysis" / "fake_H.zip").unlink()
    assert load_view_ensemble_pdb_bytes("fake_H", cache_dir=tmp_path, max_frames=10) == first


def test_load_view_ensemble_pdb_bytes_missing_bundle_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_view_ensemble_pdb_bytes("nope_A", cache_dir=tmp_path)


def _write_fake_sample(samples_dir: Path, model: str, chain: str, payload: bytes) -> Path:
    """Write a stand-in `{model}/{chain}.pdb` into a samples cache."""
    path = sample_path(model, chain, samples_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_default_samples_dir_under_home() -> None:
    assert default_samples_dir() == Path.home() / ".cache" / "premval" / "samples"


def test_available_models_lists_dirs_and_skips_scratch(tmp_path: Path) -> None:
    _write_fake_sample(tmp_path, "alphaflow_md_base", "6o2v_A", b"MODEL\n")
    _write_fake_sample(tmp_path, "esmflow_md_base", "6o2v_A", b"MODEL\n")
    (tmp_path / "_zips").mkdir()  # scratch dir must be ignored
    assert available_models(tmp_path) == ["alphaflow_md_base", "esmflow_md_base"]


def test_available_models_empty_when_dir_absent(tmp_path: Path) -> None:
    assert available_models(tmp_path / "nope") == []


def test_available_models_discovers_view_only_models(tmp_path: Path) -> None:
    # raw ensemble for one model; only a view sidecar for the other (the
    # read-only-deployment case where raw ensembles were never uploaded).
    _write_fake_sample(tmp_path, "esmdiff", "6o2v_A", b"x")
    (tmp_path / "_view" / "alphaflow_md_base").mkdir(parents=True)
    (tmp_path / "_view" / "alphaflow_md_base" / "6o2v_A.pdb").write_bytes(b"x")
    assert available_models(tmp_path) == ["alphaflow_md_base", "esmdiff"]


def test_available_chains_lists_pdb_stems(tmp_path: Path) -> None:
    _write_fake_sample(tmp_path, "alphaflow_md_base", "6o2v_A", b"x")
    _write_fake_sample(tmp_path, "alphaflow_md_base", "7ead_A", b"x")
    assert available_chains("alphaflow_md_base", tmp_path) == ["6o2v_A", "7ead_A"]
    assert available_chains("missing_model", tmp_path) == []


def test_available_chains_unions_raw_and_view_sidecars(tmp_path: Path) -> None:
    # One chain has only a raw ensemble, another only a view sidecar; both
    # should resolve, deduped and sorted.
    _write_fake_sample(tmp_path, "alphaflow_md_base", "6o2v_A", b"x")
    view_dir = tmp_path / "_view" / "alphaflow_md_base"
    view_dir.mkdir(parents=True)
    (view_dir / "6o2v_A.pdb").write_bytes(b"x")  # also has a raw ensemble
    (view_dir / "7ead_A.pdb").write_bytes(b"x")  # view-only
    assert available_chains("alphaflow_md_base", tmp_path) == ["6o2v_A", "7ead_A"]


def test_load_sample_pdb_bytes_roundtrips(tmp_path: Path) -> None:
    _write_fake_sample(tmp_path, "alphaflow_md_base", "6o2v_A", b"MODEL 1\nENDMDL\n")
    assert load_sample_pdb_bytes("alphaflow_md_base", "6o2v_A", tmp_path) == b"MODEL 1\nENDMDL\n"


def test_load_sample_pdb_bytes_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no cached sample"):
        load_sample_pdb_bytes("alphaflow_md_base", "nope_A", tmp_path)


def test_build_aligned_sample_removes_rigid_body_tumbling(tmp_path: Path) -> None:
    # Build a sample whose frames are ONE structure rotated/translated
    # differently per frame (pure rigid-body tumbling, no internal change).
    # After alignment all frames must coincide (residual ~0).
    from premval.data.samples import _build_aligned_sample

    base = make_full_atom_trajectory(n_frames=1, n_residues=6, seed=7)
    base_xyz = base.xyz[0]
    rng = np.random.default_rng(11)
    frames = []
    for _ in range(5):
        # random rotation via QR, plus a random translation
        q, _r = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        frames.append(base_xyz @ q.T + rng.standard_normal(3))
    tumbling = md.Trajectory(np.stack(frames).astype(np.float32), base.topology)
    sample_dir = tmp_path / "alphaflow_md_base"
    sample_dir.mkdir(parents=True)
    tumbling.save_pdb(str(sample_dir / "x_A.pdb"))

    ca = base.topology.select("name CA")
    ref_ca_nm = base_xyz[ca]  # align frame 0 onto the base structure's CA
    aligned_path = _build_aligned_sample("alphaflow_md_base", "x_A", ref_ca_nm, tmp_path)
    aligned = md.load(str(aligned_path)).atom_slice(ca)
    # All frames are the same structure, so post-alignment they must match frame 0.
    spread = np.sqrt(((aligned.xyz - aligned.xyz[0]) ** 2).sum(-1).mean())
    assert spread < 1e-3, f"rigid-body tumbling not removed (spread={spread:.4f} nm)"


def test_build_session_mounts_https_adapter() -> None:
    from requests.adapters import HTTPAdapter

    sess = atlas_mod._build_session()
    try:
        adapter = sess.get_adapter("https://www.dsimb.inserm.fr/")
        assert isinstance(adapter, HTTPAdapter)
        assert adapter.max_retries.total == atlas_mod._SESSION_RETRIES
    finally:
        sess.close()


def _stub_view_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy per-pair builders with marker-file writers.

    Lets the orchestration in `warm_view_caches` (chain selection, skip/overwrite,
    failure counting) be exercised without re-running the numerics those builders
    own and test elsewhere. The observables/metrics stubs write their real cache
    paths so the skip-on-rerun branch sees them.
    """
    from types import SimpleNamespace

    monkeypatch.setattr(
        samples_mod,
        "load_reference_observables",
        lambda chain, *, kind, cache_dir: SimpleNamespace(
            crystal_xyz_ca=np.zeros((1, 3), np.float32)
        ),
    )

    def _obs(model: str, chain: str, ref_ca: Any, sdir: Path, *, force: bool = False) -> Any:
        path = sample_observables_path(model, chain, sdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"npz")

    def _metrics(
        model: str, chain: str, *, cache_dir: Path, samples_dir: Path, force: bool = False
    ) -> dict[str, Any]:
        path = sample_metrics_path(model, chain, samples_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(samples_mod, "load_sample_observables", _obs)
    monkeypatch.setattr(samples_mod, "load_view_sample_pdb_bytes", lambda *a, **k: b"")
    monkeypatch.setattr(samples_mod, "load_sample_metrics", _metrics)


def test_warm_view_caches_builds_only_chains_with_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir, samples_dir = tmp_path / "atlas", tmp_path / "samples"
    _write_fake_sample(samples_dir, "alphaflow_md_base", "aaaa_A", b"x")
    _write_fake_sample(samples_dir, "alphaflow_md_base", "bbbb_B", b"x")  # no reference bundle
    bundle = cache_dir / "analysis" / "aaaa_A.zip"  # existence is all that's checked (refs stubbed)
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"PK")
    _stub_view_builders(monkeypatch)

    summary = warm_view_caches(["alphaflow_md_base"], cache_dir=cache_dir, samples_dir=samples_dir)

    assert summary == {"alphaflow_md_base": (1, 0, 0)}  # bbbb_B skipped silently (no bundle)
    assert sample_observables_path("alphaflow_md_base", "aaaa_A", samples_dir).exists()
    assert sample_metrics_path("alphaflow_md_base", "aaaa_A", samples_dir).exists()

    # Idempotent: a second pass sees the caches and skips.
    again = warm_view_caches(["alphaflow_md_base"], cache_dir=cache_dir, samples_dir=samples_dir)
    assert again == {"alphaflow_md_base": (0, 1, 0)}


def test_warm_view_caches_counts_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir, samples_dir = tmp_path / "atlas", tmp_path / "samples"
    _write_fake_sample(samples_dir, "alphaflow_md_base", "aaaa_A", b"x")
    bundle = cache_dir / "analysis" / "aaaa_A.zip"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"PK")
    _stub_view_builders(monkeypatch)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("unreadable ensemble")

    monkeypatch.setattr(samples_mod, "load_sample_observables", _boom)

    summary = warm_view_caches(["alphaflow_md_base"], cache_dir=cache_dir, samples_dir=samples_dir)

    assert summary == {"alphaflow_md_base": (0, 0, 1)}
