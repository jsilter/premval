from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import mdtraj as md


class Renderer(Protocol):
    """Renders one frame of a trajectory to a PNG.

    Implementations may pre-compute state (e.g., camera bbox) in __init__
    given the full trajectory; the protocol only constrains the per-frame call.
    """

    def render_frame(self, traj: md.Trajectory, frame_idx: int, out_png: Path) -> None: ...
