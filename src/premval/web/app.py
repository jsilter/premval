"""FastAPI app for the premval browser dashboard.

Two pages and two byte-streaming endpoints:

- `GET /` (leaderboard): placeholder table (no scored submissions yet) plus
  a sidebar of the 39 val-split chains as a browse menu.
- `GET /chain/{chain}`: NGL Viewer with playback for a chain's reference
  trajectory (subsampled to 250 frames). Accepts any chain whose bundle
  is in the cache (not restricted to the val split).
- `GET /api/chain/{chain}/topology.pdb`: raw topology PDB bytes.
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
    bundle_path,
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
                if chain in app.state.val_chains
                else f"chain {chain!r} is not cached locally"
            )
            raise HTTPException(status_code=404, detail=f"no cached bundle; {hint}")

    @app.get("/", response_class=HTMLResponse)
    def leaderboard(request: Request, _settings: SettingsDep) -> HTMLResponse:
        ctx: dict[str, Any] = {
            "chains": sorted(app.state.val_chains),
            "active_chain": None,
        }
        return templates.TemplateResponse(request, "leaderboard.html", ctx)

    @app.get("/chain/{chain}", response_class=HTMLResponse)
    def chain_page(chain: str, request: Request, settings: SettingsDep) -> HTMLResponse:
        _require_cached(chain, settings)
        ctx: dict[str, Any] = {
            "chains": sorted(app.state.val_chains),
            "active_chain": chain,
            "topology_url": f"/api/chain/{chain}/topology.pdb",
            "ensemble_url": f"/api/chain/{chain}/ensemble.pdb",
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

    return app
