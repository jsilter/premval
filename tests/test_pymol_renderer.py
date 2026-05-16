from __future__ import annotations

import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytest.importorskip("pymol")

from premval.viz.pymol_renderer import PyMOLRenderer  # noqa: E402
from premval.viz.video import render_trajectory_video  # noqa: E402


def _alanine_trace(n_frames: int = 3, n_residues: int = 10, seed: int = 0) -> md.Trajectory:
    """Build a poly-ALA trace with realistic Cα spacing (~3.8 A) and small
    per-frame jitter so PyMOL has something to draw cartoon on."""
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_residues):
        res = top.add_residue("ALA", chain)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)

    rng = np.random.default_rng(seed)
    base = np.zeros((n_residues * 4, 3), dtype=np.float32)
    for r in range(n_residues):
        x = r * 0.38
        base[r * 4 + 0] = (x - 0.1, 0.0, 0.0)
        base[r * 4 + 1] = (x, 0.0, 0.0)
        base[r * 4 + 2] = (x + 0.1, 0.0, 0.0)
        base[r * 4 + 3] = (x + 0.15, 0.1, 0.0)
    xyz = np.stack([base + rng.normal(scale=0.02, size=base.shape).astype(np.float32)
                    for _ in range(n_frames)])
    return md.Trajectory(xyz, top)


def test_pymol_renderer_writes_png(tmp_path: Path) -> None:
    traj = _alanine_trace(n_frames=2)
    renderer = PyMOLRenderer(traj, width=200, height=200, ray=False)
    out = tmp_path / "pymol_frame.png"
    renderer.render_frame(traj, 0, out)
    assert out.exists()
    # A blank 200x200 PNG is ~150-250 bytes; even a sketchy render is bigger.
    assert out.stat().st_size > 500


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_pymol_renderer_drives_full_video(tmp_path: Path) -> None:
    traj = _alanine_trace(n_frames=4)
    renderer = PyMOLRenderer(traj, width=200, height=200, ray=False)
    out = tmp_path / "pymol.mp4"
    render_trajectory_video(traj, out, renderer=renderer, fps=4)
    assert out.exists() and out.stat().st_size > 0
