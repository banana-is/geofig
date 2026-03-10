from dataclasses import dataclass
import numpy as np

@dataclass
class Ellipse:
    a: float
    b: float
    center: tuple[float, float] = (0.0, 0.0)

    def point(self, t: float) -> np.ndarray:
        cx, cy = self.center
        return np.array([
            cx + self.a * np.cos(t),
            cy + self.b * np.sin(t)
        ])

    def tangent(self, t: float) -> np.ndarray:
        return np.array([
            -self.a * np.sin(t),
             self.b * np.cos(t)
        ])

    def sample(self, n: int = 400) -> np.ndarray:
        ts = np.linspace(0, 2*np.pi, n, endpoint=True)
        return np.array([self.point(t) for t in ts])