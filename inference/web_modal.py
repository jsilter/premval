"""Modal deployment of the PREMVAL FastAPI dashboard.

Serves the leaderboard + per-chain viewer as a Modal ASGI app. Unlike the
GPU inference harnesses in this directory, this one imports ``premval``
directly (CPU only) and runs the FastAPI app `premval.web.app.create_app`
returns; no model sampling happens here, so the public URL can never trigger
a structure calculation.

Data layout (two Modal Volumes, both read-only at serve time):

- ``premval-cache`` mounted at ``/cache`` holds the contents of
  ``~/.cache/premval/atlas/``: the ``analysis/{chain}.zip`` reference bundles
  (their existence gates the chain endpoints) plus the precomputed
  ``references/analysis/{chain}.npz`` observables and
  ``references/analysis/{chain}.view.pdb`` reference playback ensembles. The
  ``.view.pdb`` is load-bearing: without it ``/ensemble.pdb`` rebuilds the
  reference per request (extract 3 replica XTCs, load ~3000 frames, subsample,
  re-serialize) at 18-42 s a hit. ``prepare-refs`` builds both. Populate once::

      premval prepare-refs --chains <all chains>   # builds .npz + .view.pdb
      modal volume put premval-cache ~/.cache/premval/atlas/ /

- ``premval-samples`` mounted at ``/samples`` carries only the precomputed
  viewer sidecars under ``samples/``: ``_view/{model}/{chain}.pdb`` playback
  ensembles, ``_observables/{model}/{chain}.npz`` and
  ``_metrics/{model}/{chain}.json`` panels. The multi-GB raw
  ``samples/{model}/{chain}.pdb`` ensembles are NOT uploaded: the viewer serves
  the sidecars, and ``available_models``/``available_chains`` discover models
  from the ``_view`` sidecars, so a model lists and renders without its raw
  ensemble on the volume. The served app does not persist its own volume
  writes, so warm the sidecars locally and push them once::

      premval prepare-samples                      # build the sidecars
      for d in _observables _metrics _view; do
          modal volume put premval-samples ~/.cache/premval/samples/$d samples/$d
      done

  The ``_aligned`` intermediate is not uploaded either: it is only the source
  the ``_view``/``_observables`` sidecars are built from, and the served app
  reads the cached outputs, never rebuilds from it.

``results/*.json`` (the committed leaderboard) is tiny, so it is baked into
the image with the source rather than mounted.

Launch
------
::

    set -a; source .env; set +a   # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
    modal serve inference/web_modal.py     # dev: hot-reloading temporary URL
    modal deploy inference/web_modal.py    # prod: persistent *.modal.run URL
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

# The web app's runtime deps: the `[web]` extra (fastapi/uvicorn/jinja2) plus
# the core scientific stack premval.data pulls in for reference observables and
# the contact recompute. Pinned loosely to match pyproject's floors.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.115",
        "uvicorn[standard]>=0.30",
        "jinja2>=3.1",
        "numpy>=2.4.5,<3",
        "scipy>=1.17.1,<2",
        "scikit-learn>=1.5,<2",
        "mdtraj>=1.11.1.post1,<2",
        "requests>=2.34.2,<3",
        "tqdm>=4.66,<5",
        "httpx>=0.28.1,<0.29",
    )
    .add_local_dir(
        local_path=str(REPO_ROOT / "src"),
        remote_path="/opt/premval/src",
    )
    .add_local_dir(
        local_path=str(REPO_ROOT / "results"),
        remote_path="/opt/premval/results",
    )
)

app = modal.App("premval-web")
cache_volume = modal.Volume.from_name("premval-cache")
samples_volume = modal.Volume.from_name("premval-samples")


@app.function(
    image=image,
    volumes={"/cache": cache_volume, "/samples": samples_volume},
    # The dashboard is read-mostly and cheap per request; allow a single
    # warm container to fan out across concurrent viewers before scaling.
    max_containers=4,
)
@modal.concurrent(max_inputs=50)
# Explicit label so the URL is `<workspace>--web.modal.run` rather than the
# auto-generated `premval-web-web` (app name + function name, doubled `web`).
@modal.asgi_app(label="web")
def web() -> object:
    """Return the PREMVAL FastAPI app wired to the mounted volumes."""
    import os
    import sys

    sys.path.insert(0, "/opt/premval/src")
    # cache_dir mirrors the local ~/.cache/premval/atlas layout: bundles at
    # /cache/analysis/{chain}.zip, observables at /cache/references/analysis/.
    os.environ["PREMVAL_CACHE_DIR"] = "/cache"
    os.environ["PREMVAL_KIND"] = "analysis"
    os.environ["PREMVAL_SAMPLES_DIR"] = "/samples/samples"
    os.environ["PREMVAL_RESULTS_DIR"] = "/opt/premval/results"

    from premval.web.app import Settings, create_app

    return create_app(Settings.from_env())
