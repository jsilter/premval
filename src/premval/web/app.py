"""FastAPI app for the premval browser dashboard.

Two pages and two byte-streaming endpoints:

- `GET /` (leaderboard): placeholder table (no scored submissions yet) plus
  a sidebar of the 39 val-split chains, each linking to its viewer page.
- `GET /chain/{chain}`: NGL Viewer with playback for a chain's reference
  trajectory (subsampled to 250 frames).
- `GET /api/chain/{chain}/topology.pdb`: raw topology PDB bytes from the
  cached ATLAS bundle.
- `GET /api/chain/{chain}/ensemble.pdb`: multi-model PDB bytes (subsampled
  reference trajectory) for NGL's trajectory player.

Settings come from `Settings.from_env()` (reads `PREMVAL_CACHE_DIR` and
`PREMVAL_KIND`) or are passed explicitly to `create_app(settings)` for
tests. NGL Viewer is loaded from the unpkg CDN; vendoring it is a
follow-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from premval.data import (
    AtlasKind,
    default_cache_dir,
    load_ensemble_pdb_bytes,
    load_topology_bytes,
    load_val_chains,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Settings:
    """Web app configuration.

    Attributes:
        cache_dir: ATLAS cache root (the directory containing
            `{kind}/{chain}.zip` bundles).
        kind: Which ATLAS payload tier to serve.
    """

    cache_dir: Path = field(default_factory=default_cache_dir)
    kind: AtlasKind = "analysis"

    @classmethod
    def from_env(cls) -> Settings:
        """Build Settings from `PREMVAL_CACHE_DIR` and `PREMVAL_KIND` env vars."""
        cache_dir_env = os.environ.get("PREMVAL_CACHE_DIR")
        kind_env: AtlasKind = os.environ.get("PREMVAL_KIND", "analysis")  # type: ignore[assignment]
        cache_dir = Path(cache_dir_env) if cache_dir_env else default_cache_dir()
        return cls(cache_dir=cache_dir, kind=kind_env)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


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
    app.state.val_chains = frozenset(load_val_chains())
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _check_known(chain: str) -> None:
        if chain not in app.state.val_chains:
            raise HTTPException(
                status_code=404,
                detail=f"chain {chain!r} not in the val split",
            )

    @app.get("/", response_class=HTMLResponse)
    def leaderboard(request: Request, _settings: SettingsDep) -> HTMLResponse:
        ctx: dict[str, Any] = {
            "chains": sorted(app.state.val_chains),
            "active_chain": None,
        }
        return templates.TemplateResponse(request, "leaderboard.html", ctx)

    @app.get("/chain/{chain}", response_class=HTMLResponse)
    def chain_page(chain: str, request: Request, _settings: SettingsDep) -> HTMLResponse:
        _check_known(chain)
        ctx: dict[str, Any] = {
            "chains": sorted(app.state.val_chains),
            "active_chain": chain,
            "topology_url": f"/api/chain/{chain}/topology.pdb",
            "ensemble_url": f"/api/chain/{chain}/ensemble.pdb",
        }
        return templates.TemplateResponse(request, "chain.html", ctx)

    @app.get("/api/chain/{chain}/topology.pdb")
    def topology_bytes(chain: str, settings: SettingsDep) -> Response:
        _check_known(chain)
        try:
            data = load_topology_bytes(chain, kind=settings.kind, cache_dir=settings.cache_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"bundle not fetched; run: premval fetch --chains {chain}",
            ) from exc
        return Response(content=data, media_type="chemical/x-pdb")

    @app.get("/api/chain/{chain}/ensemble.pdb")
    def ensemble_bytes(chain: str, settings: SettingsDep) -> Response:
        _check_known(chain)
        try:
            data = load_ensemble_pdb_bytes(chain, kind=settings.kind, cache_dir=settings.cache_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"bundle not fetched; run: premval fetch --chains {chain}",
            ) from exc
        return Response(content=data, media_type="chemical/x-pdb")

    return app
