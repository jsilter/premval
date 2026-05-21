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
from tests.test_data import _write_fake_bundle, make_full_atom_trajectory

# 6e33_A is the shortest sequence in the val split (61 residues); using a
# real val-chain name keeps the in-app `chain not in val split` 404 guard
# from firing on tests that exercise the happy path.
_REAL_CHAIN = "6e33_A"
_UNCACHED_REAL_CHAIN = "6hj6_A"


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    # Isolate the samples cache to an empty per-test dir so sample discovery
    # doesn't pick up real models from the developer's ~/.cache.
    app = create_app(
        Settings(cache_dir=tmp_path, kind="analysis", samples_dir=tmp_path / "samples")
    )
    return TestClient(app)


def test_leaderboard_lists_test_chains(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    # The browse sidebar lists the test split (the set with published samples).
    assert "6o2v_A" in body  # a known ATLAS test chain
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
    app = create_app(
        Settings(cache_dir=tmp_path, kind="analysis", samples_dir=tmp_path / "samples")
    )
    r = TestClient(app).get(f"/chain/{non_val_chain}")
    assert r.status_code == 200


def _write_placeholder_sample(samples_dir: Path, model: str, chain: str) -> None:
    """Create a sample file whose mere existence the chain page checks."""
    path = samples_dir / model / f"{chain}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def _write_real_sample(
    samples_dir: Path, model: str, chain: str, n_residues: int, n_frames: int = 6
) -> None:
    """Write a real multi-model PDB sample (so the alignment path can load it)."""
    path = samples_dir / model / f"{chain}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_full_atom_trajectory(n_frames=n_frames, n_residues=n_residues, seed=3).save_pdb(str(path))


def test_chain_page_with_sample_shows_model_picker(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    samples = tmp_path / "samples"
    _write_placeholder_sample(samples, "alphaflow_md_base", _REAL_CHAIN)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples))
    body = TestClient(app).get(f"/chain/{_REAL_CHAIN}").text
    assert 'id="modelSel"' in body
    assert "alphaflow_md_base" in body
    assert f"/api/chain/{_REAL_CHAIN}/sample/" in body


def test_chain_page_without_sample_shows_no_samples_note(client: TestClient) -> None:
    # The fixture's samples dir is empty, so the right panel shows the note
    # and no model picker.
    body = client.get(f"/chain/{_REAL_CHAIN}").text
    assert 'id="modelSel"' not in body
    assert "no model samples cached for this chain" in body


def test_sample_endpoint_returns_aligned_pdb(tmp_path: Path) -> None:
    # The toy reference bundle has 5 CA residues; the sample must match so the
    # rigid alignment (CA Kabsch) is well-defined.
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    samples = tmp_path / "samples"
    _write_real_sample(samples, "alphaflow_md_base", _REAL_CHAIN, n_residues=5)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples))
    r = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/sample/alphaflow_md_base.pdb")
    assert r.status_code == 200
    assert r.headers["content-type"] == "chemical/x-pdb"
    text = r.content.decode("ascii")
    assert text.count("\nMODEL ") + text.startswith("MODEL ") == 6  # frames preserved


def test_sample_endpoint_missing_returns_404(client: TestClient) -> None:
    r = client.get(f"/api/chain/{_REAL_CHAIN}/sample/alphaflow_md_base.pdb")
    assert r.status_code == 404
    assert "no cached sample" in r.json()["detail"]


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


def test_observables_endpoint_returns_documented_keys(tmp_path: Path) -> None:
    # Build a fake bundle and prime the reference cache for it.
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    from premval.data import load_reference_observables

    load_reference_observables(_REAL_CHAIN, cache_dir=tmp_path)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    r = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/observables.json")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "n_residues",
        "ca_indices",
        "rmsf",
        "contacts",
        "contacts_dmax_nm",
        "contacts_min_prob",
        "ellipsoids",
        "pca",
    }
    assert body["n_residues"] == 5  # _toy_chain_traj uses 5 residues
    assert len(body["ca_indices"]) == 5
    assert len(body["rmsf"]) == 5
    assert set(body["pca"].keys()) == {"mean", "modes"}
    assert len(body["pca"]["modes"]) >= 1
    for mode in body["pca"]["modes"]:
        assert len(mode["vec"]) == 5 * 3  # n_res * 3
        assert isinstance(mode["amplitude"], float)


def test_observables_endpoint_lazy_computes_on_first_request(
    client: TestClient, tmp_path: Path
) -> None:
    # The fixture builds the trajectory bundle but does NOT prime the
    # reference cache. The endpoint should compute references on first
    # request, save the .npz, and return the JSON body.
    from premval.data.references import cache_path

    refs_path = cache_path(_REAL_CHAIN, "analysis", tmp_path)
    assert not refs_path.exists()
    r = client.get(f"/api/chain/{_REAL_CHAIN}/observables.json")
    assert r.status_code == 200
    assert refs_path.exists(), "first request should write the reference .npz"
    assert "n_residues" in r.json()


def test_observables_endpoint_missing_bundle_returns_404(client: TestClient) -> None:
    # If the trajectory bundle itself isn't cached, there's no input to
    # compute references from; 404 with the same fetch-hint contract as
    # the other endpoints.
    r = client.get(f"/api/chain/{_UNCACHED_REAL_CHAIN}/observables.json")
    assert r.status_code == 404
    assert f"premval fetch --chains {_UNCACHED_REAL_CHAIN}" in r.json()["detail"]


def test_observables_endpoint_one_ellipsoid_per_residue(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(4, 4, 4))
    from premval.data import load_reference_observables

    load_reference_observables(_REAL_CHAIN, cache_dir=tmp_path)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    body = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/observables.json").json()
    assert len(body["ellipsoids"]) == body["n_residues"]
    for e in body["ellipsoids"]:
        assert len(e["lengths"]) == 3
        assert len(e["axes"]) == 3
        assert all(len(ax) == 3 for ax in e["axes"])
        # Sorted descending (principal axis first).
        assert e["lengths"][0] >= e["lengths"][1] >= e["lengths"][2]


def test_observables_endpoint_dmax_changes_contact_count(tmp_path: Path) -> None:
    # Larger distance cutoff -> at least as many contacts (monotone in dmax).
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(4, 4, 4))
    from premval.data import load_reference_observables

    load_reference_observables(_REAL_CHAIN, cache_dir=tmp_path)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    tight = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/observables.json?dmax_nm=0.4").json()
    loose = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/observables.json?dmax_nm=1.5").json()
    assert tight["contacts_dmax_nm"] == 0.4
    assert loose["contacts_dmax_nm"] == 1.5
    assert len(loose["contacts"]) >= len(tight["contacts"])


def test_observables_endpoint_dmax_out_of_range_rejected(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 3, 3))
    from premval.data import load_reference_observables

    load_reference_observables(_REAL_CHAIN, cache_dir=tmp_path)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    # Below the FastAPI Query(ge=0.3) bound.
    r = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/observables.json?dmax_nm=0.1")
    assert r.status_code == 422


def test_observables_endpoint_contacts_filtered_and_ordered(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(4, 4, 4))
    from premval.data import load_reference_observables

    load_reference_observables(_REAL_CHAIN, cache_dir=tmp_path)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis"))
    contacts = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/observables.json").json()["contacts"]
    for c in contacts:
        assert c["p"] >= 0.3
        assert c["i"] < c["j"]
        # i,i+1 is forced by the peptide bond; skip those, keep everything else.
        assert c["j"] - c["i"] >= 2
    probs = [c["p"] for c in contacts]
    assert probs == sorted(probs, reverse=True)
