"""FastAPI route tests for the premval web app.

Uses real synthesized ATLAS-style bundles (no mocking of the helpers
under test, per CODING_STANDARDS.md). `tests/test_data.py::_write_fake_bundle`
is the bundle factory.
"""

from __future__ import annotations

import json
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


def _write_results(
    results_dir: Path, model: str, split: str, rmwd_by_chain: dict[str, float]
) -> None:
    """Write a minimal `{model}.json` results payload for leaderboard tests."""
    per_target = {
        chain: {"chain": chain, "rmwd": rmwd, "rmsf_pearson": 0.5}
        for chain, rmwd in rmwd_by_chain.items()
    }
    payload = {
        "model": model,
        "split": split,
        "n_scored": len(per_target),
        "per_target": per_target,
    }
    (results_dir / f"{model}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    # Isolate the samples cache to an empty per-test dir so sample discovery
    # doesn't pick up real models from the developer's ~/.cache.
    app = create_app(
        Settings(cache_dir=tmp_path, kind="analysis", samples_dir=tmp_path / "samples")
    )
    return TestClient(app)


def test_chain_sidebar_lists_both_splits_grouped(client: TestClient) -> None:
    # The chain pages carry the browse sidebar, grouped into test and val
    # sections. (The landing page at "/" is standalone; see test_landing_page.)
    r = client.get(f"/chain/{_REAL_CHAIN}")
    assert r.status_code == 200
    body = r.text
    assert "test chains" in body
    assert "val chains" in body
    assert "6o2v_A" in body  # a known ATLAS test chain
    assert _REAL_CHAIN in body  # 6e33_A, a val chain


def test_sidebar_marks_uncached_chains(client: TestClient) -> None:
    # The fixture caches only 6e33_A; another val chain (uncached) is dimmed
    # via the `uncached` class and a fetch-hint tooltip, so missing data is
    # surfaced rather than presented as live.
    body = client.get(f"/chain/{_REAL_CHAIN}").text
    assert f"premval fetch --chains {_UNCACHED_REAL_CHAIN}" in body
    assert "uncached" in body


def test_landing_page(client: TestClient) -> None:
    # "/" is a standalone landing page: PREMVAL wordmark, a GitHub link with
    # the mark icon, and no chain sidebar.
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "PREMVAL" in body
    assert "https://github.com/jsilter/premval" in body
    assert "<svg" in body
    assert "test chains" not in body
    assert "val chains" not in body


def test_landing_leaderboard_rows(tmp_path: Path) -> None:
    # Two models scored on the same split; the lower-RMWD model ranks first and
    # its averaged metric is rendered. results_dir is isolated to tmp_path so
    # the committed results/ tree isn't read.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_results(results_dir, "good_model", "test", {"a": 1.0, "b": 3.0})  # rmwd avg 2.0
    _write_results(results_dir, "bad_model", "test", {"a": 5.0, "b": 7.0})  # rmwd avg 6.0
    app = create_app(
        Settings(
            cache_dir=tmp_path,
            kind="analysis",
            samples_dir=tmp_path / "samples",
            results_dir=results_dir,
        )
    )
    body = TestClient(app).get("/").text
    assert "good_model" in body
    assert "bad_model" in body
    assert "2.00" in body  # good_model's averaged test rmwd
    assert body.index("good_model") < body.index("bad_model")


def test_landing_uses_model_display_names_and_links(tmp_path: Path) -> None:
    # A known model key renders its proper-cased name linked to the source
    # publication, with a description tooltip.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_results(results_dir, "bioemu", "val", {"a": 4.0})
    app = create_app(
        Settings(
            cache_dir=tmp_path,
            kind="analysis",
            samples_dir=tmp_path / "samples",
            results_dir=results_dir,
        )
    )
    body = TestClient(app).get("/").text
    assert "BioEmu" in body  # display name, not the raw "bioemu" key
    assert 'href="https://doi.org/10.1101/2024.12.05.626885"' in body
    assert "equilibrium" in body  # description tooltip text
    assert "Root-mean Wasserstein" in body  # metric header tooltip
    # References section: full model citation plus the ATLAS dataset.
    assert "References" in body
    assert "Scalable emulation of protein equilibrium ensembles" in body
    assert "ATLAS: protein flexibility description" in body


def test_landing_shows_contamination_badges(tmp_path: Path) -> None:
    # Each row carries an ATLAS held-out badge: BioEmu never trained on ATLAS
    # (held out), the MD-trained flow models are held out only by a weak
    # (temporal) split. The basis is on hover.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_results(results_dir, "bioemu", "val", {"a": 4.0})
    _write_results(results_dir, "alphaflow_md_base", "test", {"a": 2.0})
    app = create_app(
        Settings(
            cache_dir=tmp_path,
            kind="analysis",
            samples_dir=tmp_path / "samples",
            results_dir=results_dir,
        )
    )
    body = TestClient(app).get("/").text
    assert 'class="badge held_out"' in body  # BioEmu held out
    assert ">held-out<" in body
    assert 'class="badge weak_holdout"' in body  # AlphaFlow-MD weak (temporal) holdout
    assert ">held-out (weak)<" in body
    assert "40% sequence identity" in body  # BioEmu basis tooltip
    assert "Data</th>" in body  # the new column header


def test_chain_page_hides_template_models(tmp_path: Path) -> None:
    # The `*_templates_*` variants clutter the picker and are excluded; a
    # normal model with a sample for the chain is still listed.
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    samples_dir = tmp_path / "samples"
    for model in ("alphaflow_md_base", "alphaflow_md_templates_base"):
        sample = samples_dir / model / f"{_REAL_CHAIN}.pdb"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text("REMARK placeholder\nEND\n", encoding="utf-8")
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples_dir))
    body = TestClient(app).get(f"/chain/{_REAL_CHAIN}").text
    assert "alphaflow_md_base" in body
    assert "templates" not in body


def test_chain_page_renders_with_ngl(client: TestClient) -> None:
    r = client.get(f"/chain/{_REAL_CHAIN}")
    assert r.status_code == 200
    body = r.text
    assert "ngl@2.3.1" in body
    assert f"/api/chain/{_REAL_CHAIN}/ensemble.pdb" in body


def test_chain_page_uncached_chain_renders_friendly_404(client: TestClient) -> None:
    # Fail gracefully: a 404 status but a friendly HTML page (in the layout),
    # not a raw JSON error.
    r = client.get("/chain/not_a_real_chain_X")
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    assert "No cached data for this chain" in r.text
    assert "is not cached locally" in r.text


def test_chain_page_val_chain_without_bundle_renders_fetch_hint(
    client: TestClient,
) -> None:
    # A val chain that hasn't been fetched: friendly 404 page with a fetch hint.
    r = client.get(f"/chain/{_UNCACHED_REAL_CHAIN}")
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    assert f"premval fetch --chains {_UNCACHED_REAL_CHAIN}" in r.text


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


def test_metadata_endpoint_serves_cached_entry(client: TestClient, tmp_path: Path) -> None:
    # Seed the metadata cache so the endpoint serves it without hitting RCSB.
    # The PDB entry id is the 4-char prefix of the chain (6e33_A -> 6e33).
    from premval.data.rcsb import EntryMetadata, entry_metadata_cache_path

    meta = EntryMetadata(
        pdb_id="6e33",
        url="https://www.rcsb.org/structure/6e33",
        title="Test entry",
        method="X-ray",
        resolution_a=1.9,
    )
    cache = entry_metadata_cache_path("6e33", tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(meta.to_dict()))

    r = client.get(f"/api/chain/{_REAL_CHAIN}/metadata.json")
    assert r.status_code == 200
    body = r.json()
    assert body["pdb_id"] == "6e33"
    assert body["title"] == "Test entry"
    assert body["resolution_a"] == 1.9


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


def test_metrics_endpoint_scores_sample_and_caches(tmp_path: Path) -> None:
    # 5-residue reference bundle + a matching 5-residue sample so the CA
    # alignment in scoring is well-defined.
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(4, 4, 4))
    samples = tmp_path / "samples"
    _write_real_sample(samples, "alphaflow_md_base", _REAL_CHAIN, n_residues=5, n_frames=8)
    # Empty results dir: averages should come back null per split, not leak the
    # developer's committed results/.
    app = create_app(
        Settings(
            cache_dir=tmp_path, kind="analysis", samples_dir=samples, results_dir=tmp_path / "res"
        )
    )

    from premval.data.samples import sample_metrics_path

    cache = sample_metrics_path("alphaflow_md_base", _REAL_CHAIN, samples)
    assert not cache.exists()
    r = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/sample/alphaflow_md_base/metrics.json")
    assert r.status_code == 200
    body = r.json()
    # The full v1 panel plus the residue/frame context the UI shows.
    assert {"rmwd", "emd_mean_rms", "emd_var_rms", "rmsf_pearson", "md_pca_w2"} <= body.keys()
    assert body["n_residues"] == 5
    assert body["n_sub_frames"] == 8
    assert body["averages"] == {"test": None, "val": None}
    assert cache.exists(), "first request should write the metrics .json"


def test_metrics_endpoint_includes_split_averages(tmp_path: Path) -> None:
    # Commit a results file for the model and confirm the endpoint folds its
    # per-split averages into the response so the table's context columns fill.
    import json

    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(4, 4, 4))
    samples = tmp_path / "samples"
    _write_real_sample(samples, "alphaflow_md_base", _REAL_CHAIN, n_residues=5)
    results = tmp_path / "res"
    results.mkdir()
    (results / "alphaflow_md_base.json").write_text(
        json.dumps(
            {
                "model": "alphaflow_md_base",
                "split": "test",
                "per_target": {
                    "aaaa_A": {"rmwd": 2.0, "rmsf_pearson": 0.8},
                    "bbbb_B": {"rmwd": 4.0, "rmsf_pearson": 0.6},
                },
                "n_scored": 2,
            }
        )
    )
    app = create_app(
        Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples, results_dir=results)
    )
    body = (
        TestClient(app)
        .get(f"/api/chain/{_REAL_CHAIN}/sample/alphaflow_md_base/metrics.json")
        .json()
    )
    assert body["averages"]["val"] is None
    assert body["averages"]["test"]["rmwd"] == 3.0  # mean(2, 4)
    assert body["averages"]["test"]["rmsf_pearson"] == pytest.approx(0.7)
    assert body["averages"]["test"]["n_chains"] == 2


def test_metrics_endpoint_missing_sample_returns_404(client: TestClient) -> None:
    r = client.get(f"/api/chain/{_REAL_CHAIN}/sample/alphaflow_md_base/metrics.json")
    assert r.status_code == 404
    assert "no cached sample" in r.json()["detail"]


def test_metrics_endpoint_nan_jaccard_serialized_as_null(tmp_path: Path) -> None:
    # Force a NaN contact Jaccard (empty union) into the metrics cache and
    # confirm the endpoint emits valid JSON null rather than a bare NaN token.
    import json

    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(4, 4, 4))
    samples = tmp_path / "samples"
    _write_real_sample(samples, "alphaflow_md_base", _REAL_CHAIN, n_residues=5)
    from premval.data.samples import sample_metrics_path

    cache = sample_metrics_path("alphaflow_md_base", _REAL_CHAIN, samples)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"rmwd": 1.0, "weak_contacts_jaccard": float("nan"), "n_residues": 5})
    )
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples))
    r = TestClient(app).get(f"/api/chain/{_REAL_CHAIN}/sample/alphaflow_md_base/metrics.json")
    assert r.status_code == 200
    assert r.json()["weak_contacts_jaccard"] is None


def test_chain_page_defaults_to_alphaflow_md(tmp_path: Path) -> None:
    # Two models, one of which sorts *before* alphaflow_md_base alphabetically
    # ('a_other' < 'alphaflow...'), so only the explicit preference can put
    # AlphaFlow-MD first. The picker's first <option> is the default selection.
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    samples = tmp_path / "samples"
    _write_placeholder_sample(samples, "a_other", _REAL_CHAIN)
    _write_placeholder_sample(samples, "alphaflow_md_base", _REAL_CHAIN)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples))
    body = TestClient(app).get(f"/chain/{_REAL_CHAIN}").text
    assert body.index("alphaflow_md_base") < body.index("a_other")


def test_chain_page_with_sample_shows_metrics_panel(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path, _REAL_CHAIN, frames_per_replica=(3, 4, 5))
    samples = tmp_path / "samples"
    _write_placeholder_sample(samples, "alphaflow_md_base", _REAL_CHAIN)
    app = create_app(Settings(cache_dir=tmp_path, kind="analysis", samples_dir=samples))
    body = TestClient(app).get(f"/chain/{_REAL_CHAIN}").text
    assert 'id="metricsPanel"' in body
    assert f"/api/chain/{_REAL_CHAIN}/sample/" in body and "metrics.json" in body
