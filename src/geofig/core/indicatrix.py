from dataclasses import dataclass
import numpy as np

@dataclass
class EllipseIndicatrix:
    a: float
    b: float

    def point(self, theta: float, scale: float = 1.0) -> np.ndarray:
        return scale * np.array([
            self.a * np.cos(theta),
            self.b * np.sin(theta)
        ])

    def sample(self, n: int = 200, scale: float = 1.0) -> np.ndarray:
        thetas = np.linspace(0, 2*np.pi, n, endpoint=True)
        return np.array([self.point(th, scale=scale) for th in thetas])