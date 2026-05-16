"""Headless PyMOL renderer for protein cartoon stills.

PyMOL is ray-traced and gives publication-style cartoons, but its Python
binding launches a single global GL context per process. We launch it
once on first instantiation, then reuse one persistent session across
every `render_frame` call to amortise the startup cost.

Camera and viewport are fixed at construction time from the bounding box
of the first frame, so the camera does not twitch as we walk the
ensemble (which would defeat the whole point of smoothness ordering).

Requires the `viz-pymol` extra (`pip install premval[viz-pymol]`).
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mdtraj as md

_PYMOL_INIT_LOCK = threading.Lock()
_PYMOL_INITIALISED = False


def _ensure_pymol_launched() -> None:
    """Initialise PyMOL exactly once per process, headlessly and quietly."""
    global _PYMOL_INITIALISED
    with _PYMOL_INIT_LOCK:
        if _PYMOL_INITIALISED:
            return
        import pymol

        pymol.finish_launching(["pymol", "-cq"])
        _PYMOL_INITIALISED = True


class PyMOLRenderer:
    """Render protein frames to PNG via a long-lived headless PyMOL session.

    Implements the `premval.viz.renderer.Renderer` protocol.
    """

    def __init__(
        self,
        traj: md.Trajectory,
        width: int = 800,
        height: int = 800,
        ray: bool = True,
        object_name: str = "frame",
    ) -> None:
        _ensure_pymol_launched()
        from pymol import cmd

        self._cmd = cmd
        self._width = width
        self._height = height
        self._ray = 1 if ray else 0
        self._object_name = object_name
        # Pin the camera using the union bounding box of all frames so the
        # view doesn't shift as different conformations come in.
        xyz = traj.xyz
        self._bbox_min = xyz.reshape(-1, 3).min(axis=0)
        self._bbox_max = xyz.reshape(-1, 3).max(axis=0)
        self._view_set = False

    def render_frame(self, traj: md.Trajectory, frame_idx: int, out_png: Path) -> None:
        cmd = self._cmd
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = Path(tmp) / "frame.pdb"
            traj[frame_idx].save_pdb(str(pdb_path))
            cmd.load(str(pdb_path), self._object_name)
            try:
                cmd.show_as("cartoon", self._object_name)
                cmd.bg_color("white")
                cmd.set("ray_opaque_background", 1)
                if not self._view_set:
                    cmd.orient(self._object_name)
                    cmd.zoom(self._object_name, buffer=2.0, complete=1)
                    self._view = cmd.get_view()
                    self._view_set = True
                else:
                    cmd.set_view(self._view)
                cmd.png(str(out_png), self._width, self._height, ray=self._ray)
            finally:
                cmd.delete(self._object_name)
