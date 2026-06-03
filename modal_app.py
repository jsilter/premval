"""Modal deployment of the premval leaderboard web app.

Serves `premval.web.app:create_app` as a Modal ASGI endpoint, reusing the repo
Dockerfile for the image so the same build runs locally (`docker run`) and on
Modal. Trajectory data lives on the existing `premval-samples` Modal Volume
(the one the inference harnesses already write samples to), mounted at /data.

Deploy:
    modal deploy modal_app.py

Populate the volume once after the first deploy (ATLAS data + the sample models
not produced by the Modal harnesses + precomputed caches); see
.scripts/upload-data-modal.sh.
"""

from __future__ import annotations

import modal

# Reuse the repo Dockerfile (installs the [web] extra, copies src + results).
# The env is restated here so the Modal container's config is self-contained
# and does not depend on the image's ENV lines surviving the build.
image = modal.Image.from_dockerfile("Dockerfile").env(
    {
        "PREMVAL_CACHE_DIR": "/data/atlas",
        "PREMVAL_SAMPLES_DIR": "/data/samples",
        "PREMVAL_RESULTS_DIR": "/app/results",
        "PREMVAL_KIND": "analysis",
        "PYTHONPATH": "/app/src",
    }
)

# Same volume the inference harnesses write to; ATLAS data is staged under
# /data/atlas on it as well (see the upload script). Mounted read-write so a
# cache miss (a chain we did not pre-warm) can compute and serve rather than
# 500; those writes are not committed, so they simply recompute next cold start.
volume = modal.Volume.from_name("premval-samples", create_if_missing=True)

app = modal.App("premval-web")


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=1.0,
    memory=2048,
    # Scale to zero when idle (a ~5-15s cold start on the next request is fine),
    # but keep a 5-minute warm tail so within-visit navigation stays snappy.
    min_containers=0,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=20)  # the chain page fires several fetches at once
@modal.asgi_app()
def web():
    from premval.web.app import create_app

    return create_app()
