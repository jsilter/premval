"""FastAPI app for the premval browser dashboard.

Two pages and several byte-streaming endpoints:

- `GET /` (landing): a standalone page (no chain sidebar) with the PREMVAL
  wordmark, a GitHub link, and a simple leaderboard of scored models ranked by
  RMWD, with a "Browse chains" link into the sidebar'd chain views.
- `GET /chain/{chain}`: NGL Viewer with playback for a chain's reference
  trajectory (subsampled to 250 frames), shown side by side with any cached
  model samples for that chain. Accepts any chain whose bundle is cached.
- `GET /api/chain/{chain}/topology.pdb`: raw topology PDB bytes.
- `GET /api/chain/{chain}/ensemble.pdb`: multi-model PDB bytes (subsampled
  reference trajectory) for NGL's trajectory player.
- `GET /api/chain/{chain}/sample/{model}.pdb`: multi-model PDB bytes of a
  generator's sample ensemble for the chain (from the samples cache).
- `GET /api/chain/{chain}/sample/{model}/metrics.json`: the v1 metric panel
  (RMWD, RMSF correlation, MD-PCA W2, contact Jaccards) scoring that sample
  against the full ATLAS reference; computed lazily and cached.
- `GET /api/chain/{chain}/observables.json`: precomputed reference overlays.

Settings come from `Settings.from_env()` (reads `PREMVAL_CACHE_DIR`,
`PREMVAL_KIND`, `PREMVAL_SAMPLES_DIR`) or are passed explicitly to
`create_app(settings)` for tests. NGL Viewer is loaded from the unpkg CDN;
vendoring it is a follow-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from numpy.typing import NDArray

from premval.data import (
    AtlasKind,
    ReferenceObservables,
    available_chains,
    available_models,
    bundle_path,
    default_cache_dir,
    default_samples_dir,
    fetch_entry_metadata,
    load_reference_observables,
    load_sample_metrics,
    load_sample_observables,
    load_test_chains,
    load_topology_bytes,
    load_val_chains,
    load_view_ensemble_pdb_bytes,
    load_view_sample_pdb_bytes,
)
from premval.leaderboard import DEFAULT_RESULTS_DIR, load_leaderboard, load_split_averages
from premval.metrics.panel import CONTACT_THRESHOLD_NM as _DEFAULT_DMAX_NM
from premval.metrics.panel import contact_probability
from premval.models import model_info

_CONTACT_MIN_PROB = 0.3
# Drop i,i+1 CA pairs: they're ~3.8 A apart because of the peptide bond,
# so they always register as in-contact regardless of dynamics and would
# clutter the overlay with a meaningless backbone necklace. Keep
# everything from |i-j| >= 2 onward (helical i,i+3/i,i+4 are real signal).
_CONTACT_MIN_SEPARATION = 2
_PCA_TOP_MODES = 3
# Model key pre-selected in the chain page's sample picker (drives both the
# right viewport and the metric panel) whenever it has a sample for the chain.
# AlphaFlow-MD is the reference generator the metric panel was ported from.
_PREFERRED_MODEL = "alphaflow_md_base"

# Frames streamed to the NGL viewer for playback. The leaderboard scores the
# full 250-frame ensembles (see `load_sample_metrics`); this smaller budget is
# purely the on-screen trajectory, where ~120 conformations already plays
# smoothly. Frame count is independent of `ca_indices` (per-atom) and of the
# separately-computed metric panel, so trimming it changes nothing the overlays
# or scores depend on.
_VIEW_MAX_FRAMES = 120

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Provenance of the reference ensembles every model is scored against.
_ATLAS_REFERENCE = {
    "url": "https://www.dsimb.inserm.fr/ATLAS/",
    "citation": (
        "Vander Meersche, Y., Cretin, G., Gheeraert, A., Gelly, J.-C., & "
        "Galochkina, T. (2024). ATLAS: protein flexibility description from "
        "atomistic molecular dynamics simulations. Nucleic Acids Research, "
        "52(D1), D384-D392."
    ),
}


def _references(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduped full citations for the leaderboard models, plus the ATLAS dataset.

    AlphaFlow and ESMFlow share one paper, so citations are deduped by URL and
    kept in leaderboard order; ATLAS (the reference data source) is appended.
    """
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        url, citation = row.get("url"), row.get("citation")
        if url and citation and url not in seen:
            seen.add(url)
            refs.append({"url": url, "citation": citation})
    refs.append(_ATLAS_REFERENCE)
    return refs


@dataclass(frozen=True)
class Settings:
    """Web app configuration.

    Attributes:
        cache_dir: ATLAS cache root (the directory containing
            `{kind}/{chain}.zip` bundles).
        kind: Which ATLAS payload tier to serve.
        samples_dir: Model-samples cache root (the directory containing
            `{model}/{chain}.pdb` ensembles served alongside the reference).
        results_dir: Committed leaderboard results root (the directory holding
            the `{model}.json` scored payloads averaged for the metric-panel
            context columns).
    """

    cache_dir: Path = field(default_factory=default_cache_dir)
    kind: AtlasKind = "analysis"
    samples_dir: Path = field(default_factory=default_samples_dir)
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)

    @classmethod
    def from_env(cls) -> Settings:
        """Build Settings from env vars (cache dir, kind, samples dir, results dir)."""
        cache_dir_env = os.environ.get("PREMVAL_CACHE_DIR")
        kind_env: AtlasKind = os.environ.get("PREMVAL_KIND", "analysis")  # type: ignore[assignment]
        samples_env = os.environ.get("PREMVAL_SAMPLES_DIR")
        results_env = os.environ.get("PREMVAL_RESULTS_DIR")
        cache_dir = Path(cache_dir_env) if cache_dir_env else default_cache_dir()
        samples_dir = Path(samples_env) if samples_env else default_samples_dir()
        results_dir = Path(results_env) if results_env else DEFAULT_RESULTS_DIR
        return cls(
            cache_dir=cache_dir, kind=kind_env, samples_dir=samples_dir, results_dir=results_dir
        )


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def _finite_or_none(value: Any) -> Any:
    """Recursively replace non-finite floats (NaN/inf) with None for valid JSON.

    Contact Jaccards are NaN when the contact union is empty; `json` emits a
    bare `NaN` token that the browser's `JSON.parse` rejects, so swap it for
    `null` (rendered as "n/a" client-side). Recurses into the nested `averages`
    sub-dicts the metrics response carries.
    """
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _serialize_contacts(prob_matrix: NDArray[np.float32]) -> list[dict[str, Any]]:
    """Filter the contact matrix and pack into the list the client expects.

    Drops |i-j| < `_CONTACT_MIN_SEPARATION` (backbone-locked pairs) and
    probabilities below `_CONTACT_MIN_PROB` (noise), then sorts by
    descending probability so the client can stop scanning the list as
    soon as the threshold slider passes a pair's value.
    """
    n_res = prob_matrix.shape[0]
    iu, ju = np.triu_indices(n_res, k=_CONTACT_MIN_SEPARATION)
    probs = prob_matrix[iu, ju]
    keep = probs >= _CONTACT_MIN_PROB
    iu, ju, probs = iu[keep], ju[keep], probs[keep]
    order = np.argsort(-probs)
    return [{"i": int(iu[k]), "j": int(ju[k]), "p": float(probs[k])} for k in order]


def _build_observables_dict(refs: ReferenceObservables, dmax_nm: float) -> dict[str, Any]:
    """Pack a `ReferenceObservables` into a browser-friendly JSON dict.

    All arrays in `ReferenceObservables` are CA-indexed (one entry per
    residue) despite the dataclass comments mentioning `n_atoms`: the
    computation upstream slices the trajectory to CA atoms before
    measuring RMSF / covariance / contact probabilities. So the client
    only needs `n_residues`-many entries; `ca_indices` tells it which
    full-topology atoms those correspond to (so it can put bfactor on
    the right atoms when coloring by flexibility).

    Heavy server-side work (ellipsoid eigendecomposition, contact-pair
    filtering) is done here so the client only has to render.

    - `contacts`: upper triangle, prob >= `_CONTACT_MIN_PROB`, sorted by
      descending probability. If `dmax_nm` matches the value the cache
      was computed at (`_DEFAULT_DMAX_NM`), reuse the cached matrix;
      otherwise recompute from `ref_xyz_ca` (still fast; one pass over
      the trajectory).
    - `ellipsoids`: per-CA eigendecomposition of the 3x3 covariance.
      Axis lengths are sqrt(eigenvalues), sorted descending (so axes[0]
      is the principal motion direction).
    - `pca.modes`: top `_PCA_TOP_MODES` components with `amplitude` =
      sqrt(explained_variance), the physically motivated sweep size.
    """
    ca = refs.ca_indices
    n_res = int(ca.shape[0])

    eigvals, eigvecs = np.linalg.eigh(refs.ref_covar)
    eigvals = eigvals[:, ::-1]
    eigvecs = eigvecs[:, :, ::-1]
    lengths = np.sqrt(np.clip(eigvals, 0.0, None))
    ellipsoids = [
        {
            "ca": int(ca[r]),
            "lengths": lengths[r].astype(float).tolist(),
            "axes": eigvecs[r].T.astype(float).tolist(),
        }
        for r in range(n_res)
    ]

    if abs(dmax_nm - _DEFAULT_DMAX_NM) < 1e-9:
        prob_matrix = refs.ref_contact_prob
    else:
        prob_matrix = contact_probability(refs.ref_xyz_ca, dmax_nm).astype(np.float32)
    contacts = _serialize_contacts(prob_matrix)

    n_modes = min(_PCA_TOP_MODES, refs.pca_components.shape[0])
    pca_modes = [
        {
            "vec": refs.pca_components[m].astype(float).tolist(),
            "amplitude": float(np.sqrt(max(refs.pca_explained_variance[m], 0.0))),
        }
        for m in range(n_modes)
    ]

    return {
        "n_residues": n_res,
        "ca_indices": ca.astype(int).tolist(),
        "rmsf": refs.ref_rmsf.astype(float).tolist(),
        "contacts": contacts,
        "contacts_dmax_nm": dmax_nm,
        "contacts_min_prob": _CONTACT_MIN_PROB,
        "ellipsoids": ellipsoids,
        "pca": {
            "mean": refs.pca_mean.astype(float).tolist(),
            "modes": pca_modes,
        },
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI factory.

    Args:
        settings: If None, reads from environment via `Settings.from_env()`
            so uvicorn's reload subprocesses can pick up the same config.

    Returns:
        Configured FastAPI app with the four routes mounted.
    """
    app = FastAPI(title="premval")
    # No app-level gzip: behind Modal's edge proxy, gzipping the large
    # multi-model PDBs (~44 MB) truncates the response mid-stream (a
    # Content-Length conflict between the app's gzip and the proxy), so the
    # viewer's structure pane silently loaded partial ensembles. Serving bodies
    # uncompressed transfers them intact; the edge can still negotiate
    # compression itself. Re-adding GZipMiddleware here reintroduces the bug.
    app.state.settings = settings or Settings.from_env()
    # `known_chains` (val U test) drives the "run premval fetch" 404 hint; the
    # sidebar is grouped by split (see `_sidebar_groups`).
    app.state.known_chains = frozenset(load_val_chains()) | frozenset(load_test_chains())
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _missing_bundle_hint(chain: str) -> str:
        """Human-readable reason a chain's bundle is missing (fetch hint vs unknown)."""
        if chain in app.state.known_chains:
            return f"run: premval fetch --chains {chain}"
        return f"chain {chain!r} is not cached locally"

    def _require_cached(chain: str, settings: Settings) -> None:
        """Raise a JSON 404 if `chain` has no cached ATLAS bundle.

        Used by the byte/JSON API endpoints, whose callers are `fetch()` in the
        page JS and already degrade gracefully. The HTML `chain_page` route
        instead renders a friendly page (see there).
        """
        if not bundle_path(settings.cache_dir, settings.kind, chain).exists():
            raise HTTPException(
                status_code=404, detail=f"no cached bundle; {_missing_bundle_hint(chain)}"
            )

    def _sidebar_groups(settings: Settings) -> list[dict[str, Any]]:
        """Build the split-grouped browse sidebar with per-chain availability.

        Two groups ("test", "val"); each chain is tagged `cached` by whether
        its ATLAS bundle is on disk, so the template can dim the ones that
        would 404 rather than presenting every chain as live.
        """
        groups: list[dict[str, Any]] = []
        for label, loader in (("test", load_test_chains), ("val", load_val_chains)):
            chains = [
                {"id": c, "cached": bundle_path(settings.cache_dir, settings.kind, c).exists()}
                for c in sorted(loader())
            ]
            groups.append({"label": label, "chains": chains})
        return groups

    @app.get("/", response_class=HTMLResponse)
    def landing(request: Request, settings: SettingsDep) -> HTMLResponse:
        groups = _sidebar_groups(settings)
        # First cached chain (test then val) so "Browse chains" lands in a live
        # viewer; fall back to the first known chain (renders the friendly page).
        first = next(
            (c["id"] for g in groups for c in g["chains"] if c["cached"]),
            next((c["id"] for g in groups for c in g["chains"]), None),
        )
        rows = load_leaderboard(settings.results_dir)
        for row in rows:
            info = model_info(row["model"])
            row["name"], row["url"], row["desc"] = info.name, info.url, info.description
            row["citation"] = info.citation
            row["contamination"] = info.contamination
            row["contamination_basis"] = info.contamination_basis
        ctx: dict[str, Any] = {
            "leaderboard": rows,
            "references": _references(rows),
            "github_url": "https://github.com/jsilter/premval",
            "browse_url": f"/chain/{first}" if first else None,
        }
        return templates.TemplateResponse(request, "landing.html", ctx)

    @app.get("/chain/{chain}", response_class=HTMLResponse)
    def chain_page(chain: str, request: Request, settings: SettingsDep) -> HTMLResponse:
        sidebar_groups = _sidebar_groups(settings)
        if not bundle_path(settings.cache_dir, settings.kind, chain).exists():
            # Fail gracefully: a friendly HTML page (still 404) inside the
            # normal layout, not a raw JSON error.
            ctx_missing: dict[str, Any] = {
                "sidebar_groups": sidebar_groups,
                "active_chain": chain,
                "chain": chain,
                "detail": _missing_bundle_hint(chain),
            }
            return templates.TemplateResponse(
                request, "chain_missing.html", ctx_missing, status_code=404
            )
        # Hide the template-conditioned variants (`*_templates_*`): they clutter
        # the picker and aren't part of the leaderboard comparison.
        sample_models = [
            model
            for model in available_models(settings.samples_dir)
            if "_templates_" not in model
            and chain in available_chains(model, settings.samples_dir)
        ]
        # Lead with the preferred generator so it is the default selection;
        # the rest keep their (alphabetical) order from `available_models`.
        sample_models.sort(key=lambda m: (m != _PREFERRED_MODEL, m))
        ctx: dict[str, Any] = {
            "sidebar_groups": sidebar_groups,
            "active_chain": chain,
            "topology_url": f"/api/chain/{chain}/topology.pdb",
            "ensemble_url": f"/api/chain/{chain}/ensemble.pdb",
            "sample_models": sample_models,
        }
        return templates.TemplateResponse(request, "chain.html", ctx)

    @app.get("/api/chain/{chain}/topology.pdb")
    def topology_bytes(chain: str, settings: SettingsDep) -> Response:
        _require_cached(chain, settings)
        data = load_topology_bytes(chain, kind=settings.kind, cache_dir=settings.cache_dir)
        return Response(content=data, media_type="chemical/x-pdb")

    @app.get("/api/chain/{chain}/ensemble.pdb")
    def ensemble_bytes(chain: str, settings: SettingsDep) -> Response:
        # Served from a precomputed view cache (built on first request, then
        # reused). Rebuilding from the bundle is expensive (extract 3 replica
        # XTCs, load ~3000 frames, subsample, re-serialize), so on the
        # read-only deployment this is warmed ahead of time by `prepare-refs`.
        _require_cached(chain, settings)
        data = load_view_ensemble_pdb_bytes(
            chain, kind=settings.kind, cache_dir=settings.cache_dir, max_frames=_VIEW_MAX_FRAMES
        )
        return Response(content=data, media_type="chemical/x-pdb")

    def _reference_ca(chain: str, settings: Settings) -> NDArray[np.float32]:
        """Reference frame-0 CA coords (nm) used to anchor sample alignment."""
        refs = load_reference_observables(chain, kind=settings.kind, cache_dir=settings.cache_dir)
        return refs.crystal_xyz_ca

    @app.get("/api/chain/{chain}/sample/{model}.pdb")
    def sample_bytes(chain: str, model: str, settings: SettingsDep) -> Response:
        # Served rigid-body aligned to the reference (frame 0 anchored to the
        # reference, each later frame to the previous), so the model is
        # oriented like the reference and doesn't tumble during playback.
        _require_cached(chain, settings)  # reference bundle needed to align against
        try:
            data = load_view_sample_pdb_bytes(
                model,
                chain,
                _reference_ca(chain, settings),
                settings.samples_dir,
                max_frames=_VIEW_MAX_FRAMES,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="chemical/x-pdb")

    @app.get("/api/chain/{chain}/sample/{model}/observables.json")
    def sample_observables_json(
        chain: str,
        model: str,
        settings: SettingsDep,
        dmax_nm: Annotated[float, Query(ge=0.3, le=2.0)] = _DEFAULT_DMAX_NM,
    ) -> JSONResponse:
        # Same overlay panel as the reference, computed from the model's own
        # (aligned) sample ensemble so the two panes are directly comparable.
        # Computed lazily on first request (can take a few seconds), then
        # cached as a .npz alongside the samples.
        _require_cached(chain, settings)
        try:
            refs = load_sample_observables(
                model, chain, _reference_ca(chain, settings), settings.samples_dir
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(_build_observables_dict(refs, dmax_nm=dmax_nm))

    @app.get("/api/chain/{chain}/sample/{model}/metrics.json")
    def sample_metrics_json(chain: str, model: str, settings: SettingsDep) -> JSONResponse:
        # The v1 metric panel (RMWD, RMSF correlation, MD-PCA W2, contact
        # Jaccards) scoring the model's sample ensemble against the full ATLAS
        # reference, same as the leaderboard. Computed lazily on first request
        # (loads the full reference; can take a few seconds), then cached as a
        # .json alongside the samples. The `averages` block carries this
        # model's per-split means (from committed results/) for context; it is
        # null per split when no results file is present. NaN Jaccards (empty
        # contact union) are sent as null so the browser's JSON.parse doesn't
        # choke.
        _require_cached(chain, settings)  # reference bundle needed to score against
        try:
            metrics = load_sample_metrics(
                model, chain, cache_dir=settings.cache_dir, samples_dir=settings.samples_dir
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        metrics["averages"] = load_split_averages(model, settings.results_dir)
        return JSONResponse(_finite_or_none(metrics))

    @app.get("/api/chain/{chain}/observables.json")
    def observables_json(
        chain: str,
        settings: SettingsDep,
        dmax_nm: Annotated[
            float,
            Query(
                ge=0.3,
                le=2.0,
                description="CA-CA distance threshold defining 'in contact', in nanometers.",
            ),
        ] = _DEFAULT_DMAX_NM,
    ) -> JSONResponse:
        # The trajectory bundle must exist (otherwise we can't compute
        # references); the reference .npz itself is built lazily on first
        # request, which can take several seconds. The client shows a
        # "computing..." indicator during the delay.
        _require_cached(chain, settings)
        refs = load_reference_observables(chain, kind=settings.kind, cache_dir=settings.cache_dir)
        return JSONResponse(_build_observables_dict(refs, dmax_nm=dmax_nm))

    @app.get("/api/chain/{chain}/metadata.json")
    def metadata_json(chain: str, settings: SettingsDep) -> JSONResponse:
        # Curated RCSB entry metadata for the chain's parent structure. The
        # PDB entry id is the 4-char prefix of the chain id (e.g. 6o2v_A ->
        # 6o2v). Fetched from data.rcsb.org on first request, then cached.
        pdb_id = chain.split("_", 1)[0]
        try:
            meta = fetch_entry_metadata(pdb_id, cache_dir=settings.cache_dir)
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502, detail=f"RCSB metadata fetch failed: {exc}"
            ) from exc
        return JSONResponse(meta.to_dict())

    return app
