"""FastAPI route tests for the premval web app.

Uses real synthesized ATLAS-style bundles (no mocking of the helpers
under test, per CODING_STANDARDS.md). `tests/test_data.py::_write_fake_bundle`
is the bundle factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from premval.web.app import Settings, create_app
from tests.test_data import _write_fake_bundle

# 6e33_A is the shortest sequence in the val split (61 residues); using a
# real val-chain name keeps the in-app `chain not in val split` 404 guard
# from firing on tests that exercise the happy path.
_REAL_CHAIN = "6e33_A"
_UNCACHED_REAL_CHAIN = "6hj6_A"


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    return TestClient(app)


def test_leaderboard_lists_val_chains(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert _REAL_CHAIN in body
    assert "no scored submissions yet" in body


def test_chain_page_renders_with_ngl(client: TestClient) -> None:
    r = client.get(f"/chain/{_REAL_CHAIN}")
    assert r.status_code == 200
    body = r.text
    assert "ngl@2.3.1" in body
    assert f"/api/chain/{_REAL_CHAIN}/ensemble.pdb" in body


def test_chain_page_uncached_chain_returns_404(client: TestClient) -> None:
    r = client.get("/chain/not_a_real_chain_X")
    assert r.status_code == 404
    assert "no cached bundle" in r.json()["detail"]


def test_chain_page_val_chain_without_bundle_returns_404_with_fetch_hint(
    client: TestClient,
) -> None:
    # A val chain that hasn't been fetched: 404 with a "premval fetch" hint.
    r = client.get(f"/chain/{_UNCACHED_REAL_CHAIN}")
    assert r.status_code == 404
    assert f"premval fetch --chains {_UNCACHED_REAL_CHAIN}" in r.json()["detail"]


def test_chain_page_accepts_non_val_cached_chain(tmp_path: Path) -> None:
    # A chain *not* in the val split, but with a cached bundle, should render.
    non_val_chain = "3tvj_I"
    assert non_val_chain not in {"6e33_A", "6hj6_A"}  # sanity: not val
    _write_fake_bundle(tmp_path, non_val_chain, frames_per_replica=(2, 2, 2))
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    r = TestClient(app).get(f"/chain/{non_val_chain}")
    assert r.status_code == 200


def test_topology_endpoint_returns_pdb_bytes(client: TestClient) -> None:
    r = client.get(f"/api/chain/{_REAL_CHAIN}/topology.pdb")
    assert r.status_code == 200
    assert r.headers["content-type"] == "chemical/x-pdb"
    head = r.content[:80].decode("ascii")
    assert head.startswith(("REMARK", "CRYST1", "MODEL", "HEADER", "ATOM"))


def test_topology_endpoint_uncached_returns_404(client: TestClient) -> None:
    r = client.get(f"/api/chain/{_UNCACHED_REAL_CHAIN}/topology.pdb")
    assert r.status_code == 404
    assert "premval fetch" in r.json()["detail"]


def test_topology_endpoint_unknown_chain_returns_404(client: TestClient) -> None:
    r = client.get("/api/chain/not_a_real_chain_X/topology.pdb")
    assert r.status_code == 404
    # Non-val chain: the detail hint should not mention `premval fetch`.
    assert "premval fetch" not in r.json()["detail"]


def test_ensemble_endpoint_returns_multi_model_pdb(client: TestClient) -> None:
    r = client.get(f"/api/chain/{_REAL_CHAIN}/ensemble.pdb")
    assert r.status_code == 200
    assert r.headers["content-type"] == "chemical/x-pdb"
    text = r.content.decode("ascii")
    # 3+4+5 = 12 frames; below the default 250 cap so all should appear.
    n_models = text.count("\nMODEL ") + text.startswith("MODEL ")
    assert n_models == 12


def test_ensemble_endpoint_uncached_returns_404(client: TestClient) -> None:
    r = client.get(f"/api/chain/{_UNCACHED_REAL_CHAIN}/ensemble.pdb")
    assert r.status_code == 404


def test_settings_from_env_reads_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREMVAL_CACHE_DIR", "/tmp/some/cache")
    monkeypatch.setenv("PREMVAL_KIND", "protein")
    s = Settings.from_env()
    assert s.cache_dir == Path("/tmp/some/cache")
    assert s.kind == "protein"


def test_cli_serve_help_does_not_require_uvicorn() -> None:
    from premval.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
