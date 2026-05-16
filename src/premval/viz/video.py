from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from premval.viz.renderer import Renderer

if TYPE_CHECKING:
    import mdtraj as md


def render_trajectory_video(
    traj: md.Trajectory,
    out_path: Path | str,
    renderer: Renderer | None = None,
    fps: int = 15,
    stride: int = 1,
) -> Path:
    """Render `traj` to an MP4 (or whatever ffmpeg infers from `out_path`).

    ffmpeg must be on PATH. If `renderer` is None, the default matplotlib
    backend is used (requires the `viz` extra).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to render videos.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if renderer is None:
        from premval.viz.matplotlib_renderer import MatplotlibRenderer

        renderer = MatplotlibRenderer(traj)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, frame_idx in enumerate(range(0, traj.n_frames, stride)):
            renderer.render_frame(traj, frame_idx, tmpdir / f"frame_{i:05d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmpdir / "frame_%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (rc={result.returncode}):\n{result.stderr.decode()}"
            )

    return out
