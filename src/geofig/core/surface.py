from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayLike = np.ndarray


@dataclass
class MongePatch:
    """
    Surface given as a graph z = f(x, y), i.e.
        r(x, y) = (x, y, f(x, y)).
    """
    f: Callable[[ArrayLike, ArrayLike], ArrayLike]

    def lift_point(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (2,):
            raise ValueError("uv must have shape (2,)")
        x, y = uv
        z = self.f(x, y)
        return np.array([x, y, z], dtype=float)

    def lift_curve(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("pts must have shape (n, 2)")

        x = pts[:, 0]
        y = pts[:, 1]
        z = self.f(x, y)
        return np.column_stack([x, y, z])

    def grid(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        nx: int = 200,
        ny: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)
        return X, Y, Z

    @staticmethod
    def mountain() -> "MongePatch":
        """
        A soft mountain-ish surface made from a few Gaussian bumps and dips.
        It is still smooth and gentle enough for overlays.
        """
        def f(x: ArrayLike, y: ArrayLike) -> ArrayLike:
            x = np.asarray(x)
            y = np.asarray(y)

            peak_1 = 1.35 * np.exp(-((x - 1.2) ** 2 / 2.8 + (y - 0.8) ** 2 / 1.8))
            peak_2 = 0.95 * np.exp(-((x + 1.8) ** 2 / 1.6 + (y + 1.0) ** 2 / 2.2))
            ridge   = 0.55 * np.exp(-((x + 0.2) ** 2 / 8.0 + (y - 1.8) ** 2 / 0.9))
            valley  = -0.75 * np.exp(-((x - 0.4) ** 2 / 2.0 + (y + 1.7) ** 2 / 1.2))
            tilt    = 0.06 * x - 0.03 * y

            return peak_1 + peak_2 + ridge + valley + tilt

        return MongePatch(f)