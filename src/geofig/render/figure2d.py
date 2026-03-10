from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LineStyle:
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


class Figure2D:
    def __init__(self, figsize: tuple[float, float] = (8, 8)) -> None:
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def add_polyline(
        self,
        pts: np.ndarray,
        *,
        style: LineStyle | None = None,
        closed: bool = False,
        **kwargs: Any,
    ) -> None:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pts must have shape (n, 2)")

        if closed and len(arr) > 0:
            arr = np.vstack([arr, arr[0]])

        merged = (style.as_kwargs() if style is not None else {}) | kwargs
        self.ax.plot(arr[:, 0], arr[:, 1], **merged)

    def set_equal(self) -> None:
        self.ax.set_aspect("equal")

    def hide_axes(self) -> None:
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(False)

    def set_limits(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

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