from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from premval.viz.pdb_input import load_pdb_set
from premval.viz.renderer import Renderer

if TYPE_CHECKING:
    import mdtraj as md


def render_trajectory_video(
    traj: md.Trajectory,
    out_path: Path | str,
    renderer: Renderer | None = None,
    fps: int = 15,
    stride: int = 1,
    frame_order: Sequence[int] | None = None,
) -> Path:
    """Render `traj` to an MP4 (or whatever ffmpeg infers from `out_path`).

    ffmpeg must be on PATH. If `renderer` is None, the default matplotlib
    backend is used (requires the `viz` extra).

    When `frame_order` is given, frames are emitted in that order and
    `stride` must be left at its default (1); the two parameters describe
    overlapping concerns and combining them silently would be confusing.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to render videos.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if frame_order is not None and stride != 1:
        raise ValueError("frame_order and stride are mutually exclusive")
    if renderer is None:
        from premval.viz.matplotlib_renderer import MatplotlibRenderer

        renderer = MatplotlibRenderer(traj)

    if frame_order is None:
        indices: Sequence[int] = list(range(0, traj.n_frames, stride))
    else:
        indices = list(frame_order)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, frame_idx in enumerate(indices):
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


def render_pdbs_video(
    pdb_paths: Sequence[Path | str],
    out_path: Path | str,
    *,
    renderer: Renderer | None = None,
    fps: int = 15,
    reorder: bool = False,
) -> Path:
    """Load a list of single-structure PDBs and render them as a video.

    Args:
        pdb_paths: Ordered sequence of single-frame PDB file paths.
            Topologies must match across files (see `load_pdb_set`).
        out_path: Output video path; ffmpeg infers the format.
        renderer: Per-frame renderer. Defaults to the matplotlib backend.
        fps: Video framerate.
        reorder: If True, permute frames via
            `premval.viz.ordering.smooth_order` so adjacent frames are as
            similar as possible. Off by default so an already-meaningful
            input order is preserved.

    Returns:
        The resolved `out_path` after the video has been written.
    """
    traj = load_pdb_set(pdb_paths)
    frame_order: Sequence[int] | None = None
    if reorder:
        from premval.viz.ordering import smooth_order

        frame_order = smooth_order(traj)
    return render_trajectory_video(
        traj, out_path, renderer=renderer, fps=fps, frame_order=frame_order
    )
