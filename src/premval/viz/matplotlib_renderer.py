from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    import mdtraj as md


class MatplotlibRenderer:
    """CA-trace renderer. Ugly, no protein-viz deps, fine for end-to-end smoke."""

    def __init__(
        self,
        traj: md.Trajectory,
        dpi: int = 100,
        figsize: tuple[float, float] = (6.0, 6.0),
    ) -> None:
        ca_indices = traj.topology.select("name CA")
        if len(ca_indices) == 0:
            raise ValueError("Trajectory has no CA atoms to render.")
        xyz = traj.xyz[:, ca_indices, :]
        self._ca_indices = ca_indices
        self._xlim = (float(xyz[..., 0].min()), float(xyz[..., 0].max()))
        self._ylim = (float(xyz[..., 1].min()), float(xyz[..., 1].max()))
        self._zlim = (float(xyz[..., 2].min()), float(xyz[..., 2].max()))
        self._dpi = dpi
        self._figsize = figsize

    def render_frame(self, traj: md.Trajectory, frame_idx: int, out_png: Path) -> None:
        coords = traj.xyz[frame_idx, self._ca_indices, :]
        fig = plt.figure(figsize=self._figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=2)
        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)
        ax.set_zlim(*self._zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        fig.savefig(out_png, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)
