"""FastAPI app for the premval browser dashboard.

Two pages and several byte-streaming endpoints:

- `GET /` (leaderboard): placeholder table (no scored submissions yet) plus
  a sidebar of the test-split chains as a browse menu (the split that has
  published model samples to compare against).
- `GET /chain/{chain}`: NGL Viewer with playback for a chain's reference
  trajectory (subsampled to 250 frames), shown side by side with any cached
  model samples for that chain. Accepts any chain whose bundle is cached.
- `GET /api/chain/{chain}/topology.pdb`: raw topology PDB bytes.
- `GET /api/chain/{chain}/ensemble.pdb`: multi-model PDB bytes (subsampled
  reference trajectory) for NGL's trajectory player.
- `GET /api/chain/{chain}/sample/{model}.pdb`: multi-model PDB bytes of a
  generator's sample ensemble for the chain (from the samples cache).
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
from typing import Annotated, Any, cast

import numpy as np
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
    load_ensemble_pdb_bytes,
    load_reference_observables,
    load_sample_observables,
    load_sample_pdb_bytes,
    load_test_chains,
    load_topology_bytes,
    load_val_chains,
)
from premval.metrics.panel import CONTACT_THRESHOLD_NM as _DEFAULT_DMAX_NM

_CONTACT_MIN_PROB = 0.3
# Drop i,i+1 CA pairs: they're ~3.8 A apart because of the peptide bond,
# so they always register as in-contact regardless of dynamics and would
# clutter the overlay with a meaningless backbone necklace. Keep
# everything from |i-j| >= 2 onward (helical i,i+3/i,i+4 are real signal).
_CONTACT_MIN_SEPARATION = 2
_PCA_TOP_MODES = 3

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Settings:
    """Web app configuration.

    Attributes:
        cache_dir: ATLAS cache root (the directory containing
            `{kind}/{chain}.zip` bundles).
        kind: Which ATLAS payload tier to serve.
        samples_dir: Model-samples cache root (the directory containing
            `{model}/{chain}.pdb` ensembles served alongside the reference).
    """

    cache_dir: Path = field(default_factory=default_cache_dir)
    kind: AtlasKind = "analysis"
    samples_dir: Path = field(default_factory=default_samples_dir)

    @classmethod
    def from_env(cls) -> Settings:
        """Build Settings from env vars (cache dir, kind, samples dir)."""
        cache_dir_env = os.environ.get("PREMVAL_CACHE_DIR")
        kind_env: AtlasKind = os.environ.get("PREMVAL_KIND", "analysis")  # type: ignore[assignment]
        samples_env = os.environ.get("PREMVAL_SAMPLES_DIR")
        cache_dir = Path(cache_dir_env) if cache_dir_env else default_cache_dir()
        samples_dir = Path(samples_env) if samples_env else default_samples_dir()
        return cls(cache_dir=cache_dir, kind=kind_env, samples_dir=samples_dir)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def _compute_contact_prob(xyz_ca: NDArray[np.float32], dmax_nm: float) -> NDArray[np.float32]:
    """Recompute the (n_res, n_res) contact-probability matrix at a new threshold.

    Streams frame-by-frame so the intermediate stays at
    `(n_res, n_res, 3)` instead of materializing
    `(n_frames, n_res, n_res, 3)` (which OOMs at ~400 residues).
    """
    n_res = xyz_ca.shape[1]
    count = np.zeros((n_res, n_res), dtype=np.float32)
    for frame in xyz_ca:
        d = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=-1)
        count += (d < dmax_nm).astype(np.float32)
    return cast("NDArray[np.float32]", (count / xyz_ca.shape[0]).astype(np.float32))


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
        prob_matrix = _compute_contact_prob(refs.ref_xyz_ca, dmax_nm)
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
    app.state.settings = settings or Settings.from_env()
    # The browse sidebar lists the test split: that is the set AlphaFlow /
    # ESMFlow published samples for, so it is where the side-by-side
    # reference-vs-samples comparison is available. `known_chains` (val U
    # test) still drives the "run premval fetch" 404 hint.
    app.state.browse_chains = sorted(load_test_chains())
    app.state.known_chains = frozenset(load_val_chains()) | frozenset(load_test_chains())
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _require_cached(chain: str, settings: Settings) -> None:
        """Raise 404 if `chain` has no cached ATLAS bundle.

        The viewer accepts any chain ATLAS knows about (not restricted to
        the val split). The val split is meaningful for the leaderboard
        sidebar, but for the viewer the only thing that matters is whether
        we have the bytes on disk.
        """
        if not bundle_path(settings.cache_dir, settings.kind, chain).exists():
            hint = (
                f"run: premval fetch --chains {chain}"
                if chain in app.state.known_chains
                else f"chain {chain!r} is not cached locally"
            )
            raise HTTPException(status_code=404, detail=f"no cached bundle; {hint}")

    @app.get("/", response_class=HTMLResponse)
    def leaderboard(request: Request, _settings: SettingsDep) -> HTMLResponse:
        ctx: dict[str, Any] = {
            "chains": app.state.browse_chains,
            "active_chain": None,
        }
        return templates.TemplateResponse(request, "leaderboard.html", ctx)

    @app.get("/chain/{chain}", response_class=HTMLResponse)
    def chain_page(chain: str, request: Request, settings: SettingsDep) -> HTMLResponse:
        _require_cached(chain, settings)
        sample_models = [
            model
            for model in available_models(settings.samples_dir)
            if chain in available_chains(model, settings.samples_dir)
        ]
        ctx: dict[str, Any] = {
            "chains": app.state.browse_chains,
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
        _require_cached(chain, settings)
        data = load_ensemble_pdb_bytes(chain, kind=settings.kind, cache_dir=settings.cache_dir)
        return Response(content=data, media_type="chemical/x-pdb")

    @app.get("/api/chain/{chain}/sample/{model}.pdb")
    def sample_bytes(chain: str, model: str, settings: SettingsDep) -> Response:
        try:
            data = load_sample_pdb_bytes(model, chain, settings.samples_dir)
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
        # Same overlay panel as the reference, but computed from the model's
        # own sample ensemble so the two panes are directly comparable.
        # Computed lazily on first request (can take a few seconds), then
        # cached as a .npz alongside the samples.
        try:
            refs = load_sample_observables(model, chain, settings.samples_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(_build_observables_dict(refs, dmax_nm=dmax_nm))

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

    return app
