from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geofig import Figure2D, LineStyle
from geofig import Geodesic2D, GeodesicSpray2D


@dataclass
class GeodesicSprayScene:
    spray: GeodesicSpray2D
    s_min: float = -2.0
    s_max: float = 2.0
    n_samples: int = 300
    front_radii: list[float] = field(default_factory=list)
    highlight_index: int | None = None
    show_base_point: bool = True
    title: str = "Geodesic spray"

    geodesic_style: LineStyle = field(default_factory=lambda: LineStyle(color="0.35", linewidth=1.3, alpha=0.65))
    front_style: LineStyle = field(default_factory=lambda: LineStyle(color="0.1", linewidth=2.2, alpha=0.95))
    highlight_style: LineStyle = field(default_factory=lambda: LineStyle(color="0.0", linewidth=2.8, alpha=1.0))

    def draw(self, fig: Figure2D) -> Figure2D:
        for i, geodesic in enumerate(self.spray.geodesics):
            style = self.highlight_style if self.highlight_index == i else self.geodesic_style
            fig.add_polyline(geodesic.sample(self.s_min, self.s_max, self.n_samples), style=style)

        for r in self.front_radii:
            pts = self.spray.front_at(r)
            fig.add_polyline(pts, style=self.front_style, closed=True)
            fig.ax.text(
                pts[0, 0] + 0.05,
                pts[0, 1] + 0.05,
                fr"$s={r}$",
                fontsize=10,
            )

        if self.show_base_point:
            p = self.spray.base_point
            fig.ax.scatter([p[0]], [p[1]], s=45)
            fig.ax.text(p[0] + 0.07, p[1] + 0.08, r"$p$", fontsize=11)

        fig.ax.axhline(0.0, linewidth=0.8)
        fig.ax.axvline(0.0, linewidth=0.8)
        fig.ax.grid(True, alpha=0.25)
        fig.ax.set_title(self.title)
        fig.ax.set_xlabel("x")
        fig.ax.set_ylabel("y")
        fig.set_equal()
        return fig

    def make_figure(self, figsize: tuple[float, float] = (8, 8)) -> Figure2D:
        fig = Figure2D(figsize=figsize)
        return self.draw(fig)

    def save(self, path: str | Path, *, figsize: tuple[float, float] = (8, 8), dpi: int = 180) -> Path:
        fig = self.make_figure(figsize=figsize)
        fig.tight_layout()
        fig.save(path, dpi=dpi)
        return Path(path)
