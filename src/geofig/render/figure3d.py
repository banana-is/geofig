from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass(frozen=True)
class LineStyle3D:
    color: str = "black"
    linewidth: float = 2.0
    linestyle: str = "-"
    alpha: float = 1.0
    label: str | None = None

    def as_kwargs(self) -> dict[str, Any]:
        out = {
            "color": self.color,
            "linewidth": self.linewidth,
            "linestyle": self.linestyle,
            "alpha": self.alpha,
        }
        if self.label is not None:
            out["label"] = self.label
        return out


class Figure3D:
    def __init__(self, figsize: tuple[float, float] = (11, 8)) -> None:
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")

    def add_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        cmap: str = "Greys",
        alpha: float = 0.35,
        linewidth: float = 0.0,
        antialiased: bool = True,
        zorder: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.ax.plot_surface(
            X, Y, Z,
            cmap=cmap,
            alpha=alpha,
            linewidth=linewidth,
            antialiased=antialiased,
            zorder=zorder,
            **kwargs,
        )

    def add_polyline(
        self,
        pts: np.ndarray,
        *,
        style: LineStyle3D | None = None,
        closed: bool = False,
        **kwargs: Any,
    ) -> None:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("pts must have shape (n, 3)")

        if closed and len(arr) > 0:
            arr = np.vstack([arr, arr[0]])

        merged = (style.as_kwargs() if style is not None else {}) | kwargs
        self.ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], **merged)

    def add_points(self, pts: np.ndarray, **kwargs: Any) -> None:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("pts must have shape (n, 3)")
        self.ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], **kwargs)

    def set_view(self, elev: float = 28, azim: float = -58) -> None:
        self.ax.view_init(elev=elev, azim=azim)

    def set_limits(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        zlim: tuple[float, float] | None = None,
    ) -> None:
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        if zlim is not None:
            self.ax.set_zlim(*zlim)

    def hide_axes(self) -> None:
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_zlabel("")
        self.ax.set_axis_off()

    def add_legend(self, **kwargs: Any) -> None:
        self.ax.legend(**kwargs)

    def tight_layout(self) -> None:
        self.fig.tight_layout()

    def save(self, path: str | Path, **kwargs: Any) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, bbox_inches="tight", **kwargs)

    def show(self) -> None:
        plt.show()