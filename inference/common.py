"""Shared helpers for the inference/ GPU harnesses.

The importable ``premval`` package is CPU-only and never imports this module.
This file lives in ``inference/`` alongside the harness scripts (``str2str_run``,
``bioemu_run``, ``confdiff_run``) and is imported by them as a sibling module
(``from common import ...``) when a script is run as ``python inference/<x>.py``.

Right now it holds the run telemetry that every harness records: per-chain wall
time and, when a CUDA device is present, peak and time-averaged GPU memory. The
numbers are the same ones a wandb/Modal system panel would surface, captured
locally so the harnesses stay dependency-free (the package constraint) and so a
manual GPU run on the user's own box needs no account or network. Each chain's
telemetry is written as a JSON sidecar next to its sample PDB
(``{model}/{chain}.telemetry.json``), which a later dashboard reader can
aggregate without re-running anything.

torch is imported lazily so this module, and the harness ``--self-test`` paths,
work on a CPU-only ``premval`` install with no torch present.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

_BYTES_PER_MB = 1024 * 1024
_DEFAULT_POLL_INTERVAL_S = 0.5


def _cuda() -> ModuleType | None:
    """Return ``torch.cuda`` if a CUDA device is usable, else ``None``.

    Importing torch here (not at module top) keeps the harnesses importable on a
    CPU-only install; returning ``None`` lets the telemetry degrade to wall-time
    only when there is no GPU.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


@dataclasses.dataclass(frozen=True)
class SampleTelemetry:
    """Timing and GPU-memory record for sampling one chain's ensemble.

    Attributes:
        chain: Chain identifier, e.g. ``6o2v_A``.
        n_samples: Number of conformations generated for the chain.
        wall_seconds: Wall-clock time spent in the sampling block.
        seconds_per_sample: ``wall_seconds / n_samples`` (0 if ``n_samples`` is 0).
        device: CUDA device name, or ``"cpu"`` when no GPU was used.
        gpu_peak_mb: Peak allocated GPU memory during sampling (``None`` on CPU).
        gpu_mean_mb: Time-averaged allocated GPU memory (``None`` on CPU).
        gpu_poll_count: Number of background memory polls taken (0 on CPU or when
            sampling finished within one poll interval).
    """

    chain: str
    n_samples: int
    wall_seconds: float
    seconds_per_sample: float
    device: str
    gpu_peak_mb: float | None
    gpu_mean_mb: float | None
    gpu_poll_count: int

    def summary(self) -> str:
        """One-line human-readable summary for stdout."""
        line = (
            f"telemetry {self.chain}: n={self.n_samples} "
            f"wall={self.wall_seconds:.1f}s ({self.seconds_per_sample:.3f}s/sample) "
            f"device={self.device}"
        )
        if self.gpu_peak_mb is not None:
            line += f" gpu_peak={self.gpu_peak_mb:.0f}MB gpu_mean={self.gpu_mean_mb:.0f}MB"
        return line


class _GpuMemoryPoller(threading.Thread):
    """Background thread sampling ``torch.cuda.memory_allocated`` at a fixed rate.

    Gives the time-averaged GPU-memory figure (peak comes for free from
    ``max_memory_allocated``). A daemon thread so it can never block process
    exit; it stops promptly when ``stop`` sets the event.
    """

    def __init__(self, cuda: ModuleType, interval_s: float) -> None:
        super().__init__(daemon=True)
        self._cuda = cuda
        self._interval_s = interval_s
        self._stop = threading.Event()
        self.samples_bytes: list[int] = []

    def run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.samples_bytes.append(self._cuda.memory_allocated())

    def stop(self) -> None:
        self._stop.set()
        self.join()


@contextmanager
def track_sample(
    chain: str,
    n_samples: int,
    *,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
) -> Iterator[list[SampleTelemetry]]:
    """Time a sampling block and record GPU memory, yielding a one-slot sink.

    Use as::

        with track_sample(chain, n_samples) as sink:
            ...heavy GPU sampling...
        telemetry = sink[0]

    The yielded list is empty inside the block and holds exactly one
    `SampleTelemetry` once the block exits (so the caller can read the result
    after the `with`). On a CPU-only run the GPU fields are ``None`` and only
    wall time is recorded.

    Args:
        chain: Chain identifier being sampled.
        n_samples: Conformations the block will generate (for the per-sample rate).
        poll_interval_s: Seconds between background GPU-memory polls.

    Yields:
        A list that the context manager fills with one `SampleTelemetry` on exit.
    """
    sink: list[SampleTelemetry] = []
    cuda = _cuda()
    poller: _GpuMemoryPoller | None = None
    if cuda is not None:
        cuda.reset_peak_memory_stats()
        poller = _GpuMemoryPoller(cuda, poll_interval_s)
        poller.start()

    start = time.perf_counter()
    try:
        yield sink
    finally:
        wall = time.perf_counter() - start
        sink.append(_finalize(chain, n_samples, wall, cuda, poller))


def _finalize(
    chain: str,
    n_samples: int,
    wall: float,
    cuda: ModuleType | None,
    poller: _GpuMemoryPoller | None,
) -> SampleTelemetry:
    """Stop the poller (if any) and assemble the `SampleTelemetry`."""
    per_sample = wall / n_samples if n_samples else 0.0
    if cuda is None or poller is None:
        return SampleTelemetry(
            chain=chain,
            n_samples=n_samples,
            wall_seconds=wall,
            seconds_per_sample=per_sample,
            device="cpu",
            gpu_peak_mb=None,
            gpu_mean_mb=None,
            gpu_poll_count=0,
        )

    poller.stop()
    peak_mb = cuda.max_memory_allocated() / _BYTES_PER_MB
    polls = poller.samples_bytes
    # Sampling can finish within one poll interval; fall back to peak so the
    # mean is never null when a GPU was in use.
    mean_mb = (sum(polls) / len(polls) / _BYTES_PER_MB) if polls else peak_mb
    return SampleTelemetry(
        chain=chain,
        n_samples=n_samples,
        wall_seconds=wall,
        seconds_per_sample=per_sample,
        device=cuda.get_device_name(),
        gpu_peak_mb=peak_mb,
        gpu_mean_mb=mean_mb,
        gpu_poll_count=len(polls),
    )


class TelemetryLogger:
    """Forwards per-chain telemetry to a Weights & Biases run, or no-ops.

    Constructed by `wandb_run`. When wandb logging is disabled (the common
    case: no ``WANDB_PROJECT``), it holds ``None`` and every `log` call is a
    cheap no-op, so the harness loop is wandb-agnostic. The JSON sidecars remain
    the source of truth either way; wandb just adds live charts and, for free, a
    GPU/system-metrics panel while the run is active.
    """

    def __init__(self, run: Any | None) -> None:
        self._run = run

    def log(self, telemetry: SampleTelemetry) -> None:
        """Log one chain's telemetry as a wandb step (no-op if disabled)."""
        if self._run is None:
            return
        metrics: dict[str, Any] = {
            "chain": telemetry.chain,
            "wall_seconds": telemetry.wall_seconds,
            "seconds_per_sample": telemetry.seconds_per_sample,
        }
        if telemetry.gpu_peak_mb is not None:
            metrics["gpu_peak_mb"] = telemetry.gpu_peak_mb
            metrics["gpu_mean_mb"] = telemetry.gpu_mean_mb
        self._run.log(metrics)


@contextmanager
def wandb_run(
    model: str,
    split: str,
    config: dict[str, Any] | None = None,
) -> Iterator[TelemetryLogger]:
    """Open a wandb run for a harness invocation if ``WANDB_PROJECT`` is set.

    Optional and off by default. Logging is enabled only when ``WANDB_PROJECT``
    is in the environment; wandb then authenticates the usual way (``WANDB_API_KEY``
    or a prior ``wandb login``). When disabled, yields a no-op `TelemetryLogger`
    so callers need no conditional. wandb is imported lazily here so the harnesses
    (and ``--self-test``) keep running on an install without wandb present.

    Args:
        model: Out-model / samples-cache key, used in the run name.
        split: ATLAS split being sampled, used in the run name.
        config: Optional run config recorded as wandb hyperparameters.

    Yields:
        A `TelemetryLogger` (live when enabled, no-op otherwise).
    """
    if not os.environ.get("WANDB_PROJECT"):
        yield TelemetryLogger(None)
        return

    import wandb

    run = wandb.init(name=f"{model}-{split}", config=config or {}, job_type="inference")
    try:
        yield TelemetryLogger(run)
    finally:
        wandb.finish()


def telemetry_path(sample_pdb: Path) -> Path:
    """Sidecar path for a sample PDB: ``{chain}.pdb`` -> ``{chain}.telemetry.json``."""
    return sample_pdb.with_suffix(".telemetry.json")


def write_telemetry(sample_pdb: Path, telemetry: SampleTelemetry) -> Path:
    """Write the telemetry JSON sidecar next to ``sample_pdb``; return its path."""
    path = telemetry_path(sample_pdb)
    path.write_text(json.dumps(dataclasses.asdict(telemetry), indent=2) + "\n", encoding="utf-8")
    return path
