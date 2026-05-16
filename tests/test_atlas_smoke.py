"""End-to-end smoke against a real ATLAS bundle.

Skipped unless the user has run `fetch_val_split()` (or otherwise placed
a 6cka_B bundle in the default cache) and has ffmpeg + the viz-pymol
extra installed. The test extracts a handful of frames from the bundle's
R1 replicate, writes them as single-frame PDBs, and exercises the full
`render_pdbs_video(reorder=True)` pipeline with the PyMOL renderer.

Marked separately from the unit suite via the `slow` keyword; opt in
with `pytest -k atlas_smoke` or run the full suite explicitly.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import mdtraj as md
import pytest

from premval.data import default_cache_dir
from premval.viz.video import render_pdbs_video

pytest.importorskip("pymol")

_BUNDLE_CHAIN = "6cka_B"
_BUNDLE_PATH = default_cache_dir() / "analysis" / f"{_BUNDLE_CHAIN}.zip"


pytestmark = [
    pytest.mark.skipif(not _BUNDLE_PATH.exists(), reason=f"{_BUNDLE_PATH} not cached"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
]


def _extract_frames(bundle: Path, dest: Path, n: int = 8) -> list[Path]:
    """Extract `n` evenly-spaced frames from the bundle's R1 trajectory."""
    with zipfile.ZipFile(bundle) as zf:
        zf.extract(f"{_BUNDLE_CHAIN}.pdb", dest)
        zf.extract(f"{_BUNDLE_CHAIN}_R1.xtc", dest)

    top = md.load(str(dest / f"{_BUNDLE_CHAIN}.pdb"))
    traj = md.load(str(dest / f"{_BUNDLE_CHAIN}_R1.xtc"), top=top.topology)

    stride = max(1, traj.n_frames // n)
    indices = list(range(0, traj.n_frames, stride))[:n]
    pdb_dir = dest / "pdbs"
    pdb_dir.mkdir()
    paths: list[Path] = []
    for frame_idx in indices:
        path = pdb_dir / f"{_BUNDLE_CHAIN}_{frame_idx:05d}.pdb"
        traj[frame_idx].save_pdb(str(path))
        paths.append(path)
    return paths


def test_atlas_bundle_renders_smooth_video(tmp_path: Path) -> None:
    from premval.viz.pymol_renderer import PyMOLRenderer

    pdb_paths = _extract_frames(_BUNDLE_PATH, tmp_path, n=6)

    from premval.viz.pdb_input import load_pdb_set

    loaded = load_pdb_set(pdb_paths)
    renderer = PyMOLRenderer(loaded, width=200, height=200, ray=False)

    out = tmp_path / "atlas_smooth.mp4"
    render_pdbs_video(pdb_paths, out, renderer=renderer, fps=4, reorder=True)
    assert out.exists() and out.stat().st_size > 1000
