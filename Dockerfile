# Portable container image for the premval dashboard (FastAPI + NGL chain viewer).
#
# The canonical hosted deployment is Modal (see inference/web_modal.py); this
# image runs the same `premval.web.app:create_app` on any host, or locally:
#
#   docker build -t premval-web .
#   docker run -p 8080:8080 -v ~/.cache/premval:/data premval-web
#
# Trajectory data is NOT baked in: mount the premval cache (the `atlas/` and
# `samples/` trees, with their precomputed reference and viewer caches) at /data.
# With a writable volume the app also lazily computes and writes any missing
# observables / metrics / view PDBs on first request.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install the package with the [web] extra. numpy / scipy / scikit-learn /
# mdtraj all publish manylinux wheels for cp312, so no compiler toolchain is
# needed (pymol was dropped from core deps, which is what made this slim).
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[web]"

# Committed leaderboard payloads (small; PREMVAL_RESULTS_DIR points here).
COPY results ./results

# Cache roots live on the mounted volume (/data); the app reads bundles from
# them and lazily writes computed observables/metrics/view PDBs back.
ENV PREMVAL_CACHE_DIR=/data/atlas \
    PREMVAL_SAMPLES_DIR=/data/samples \
    PREMVAL_RESULTS_DIR=/app/results \
    PREMVAL_KIND=analysis

EXPOSE 8080

# Single worker: the metric / observable computations are CPU- and
# memory-heavy and are cached to the volume after the first hit, so one worker
# keeps peak memory bounded (matters under shared-cpu-1x / 2 GB).
CMD ["uvicorn", "premval.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
