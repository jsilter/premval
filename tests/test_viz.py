from __future__ import annotations

import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from premval.viz.matplotlib_renderer import MatplotlibRenderer
from premval.viz.video import render_trajectory_video


def _toy_traj(n_frames: int = 5, n_residues: int = 10, seed: int = 0) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("CA", md.element.carbon, res)
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_frames, n_residues, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def test_matplotlib_renderer_writes_png(tmp_path: Path) -> None:
    traj = _toy_traj()
    renderer = MatplotlibRenderer(traj)
    out = tmp_path / "frame.png"
    renderer.render_frame(traj, 0, out)
    assert out.exists() and out.stat().st_size > 0


def test_matplotlib_renderer_rejects_no_ca(tmp_path: Path) -> None:
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("HOH", chain)
    top.add_atom("O", md.element.oxygen, res)
    xyz = np.zeros((1, 1, 3), dtype=np.float32)
    traj = md.Trajectory(xyz, top)
    with pytest.raises(ValueError, match="no CA atoms"):
        MatplotlibRenderer(traj)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_trajectory_video(tmp_path: Path) -> None:
    traj = _toy_traj(n_frames=10)
    out = tmp_path / "out.mp4"
    render_trajectory_video(traj, out, fps=10)
    assert out.exists() and out.stat().st_size > 0
