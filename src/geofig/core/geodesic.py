from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayLike2D = np.ndarray | tuple[float, float] | list[float]


def _as_point(x: ArrayLike2D) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D point/vector with shape (2,)")
    return arr


@dataclass
class Geodesic2D:
    """A parametrized planar geodesic/curve s ↦ gamma(s).

    The class is intentionally lightweight: it stores a position map and,
    optionally, a velocity map. This makes it usable both for analytic
    geodesics (Euclidean lines, known model spaces) and later for numerically
    integrated geodesics where gamma(s) may come from interpolation.
    """

    position: Callable[[float], ArrayLike2D]
    velocity_map: Callable[[float], ArrayLike2D] | None = None
    name: str | None = None

    def point(self, s: float) -> np.ndarray:
        return _as_point(self.position(float(s)))

    def velocity(self, s: float, h: float = 1e-6) -> np.ndarray:
        if self.velocity_map is not None:
            return _as_point(self.velocity_map(float(s)))

        s = float(s)
        return (self.point(s + h) - self.point(s - h)) / (2.0 * h)

    def speed(self, s: float) -> float:
        v = self.velocity(s)
        return float(np.linalg.norm(v))

    def sample(self, s_min: float, s_max: float, n: int = 400) -> np.ndarray:
        ss = np.linspace(s_min, s_max, n)
        return np.array([self.point(s) for s in ss])

    @classmethod
    def affine(cls, base_point: ArrayLike2D, direction: ArrayLike2D, *, name: str | None = None) -> "Geodesic2D":
        """Create gamma(s) = p + s v.

        In the Euclidean plane, these are exactly the unit-speed geodesics when
        ||v|| = 1.
        """
        p = _as_point(base_point)
        v = _as_point(direction)

        def position(s: float) -> np.ndarray:
            return p + s * v

        def velocity_map(s: float) -> np.ndarray:
            return v

        return cls(position=position, velocity_map=velocity_map, name=name)

    @classmethod
    def from_angle(cls, base_point: ArrayLike2D, theta: float, *, unit_speed: bool = True, name: str | None = None) -> "Geodesic2D":
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        if not unit_speed:
            raise NotImplementedError("Only unit-speed angle geodesics are implemented in this constructor.")
        return cls.affine(base_point=base_point, direction=direction, name=name)


@dataclass
class GeodesicSpray2D:
    """A family of geodesics issued from one base point.

    This is the object you want for exponential-map style pictures:
    - fix one geodesic and vary s  -> one trajectory
    - fix s and vary the geodesic  -> one front / distance sphere
    """

    base_point: np.ndarray
    geodesics: list[Geodesic2D]
    parameters: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.base_point = _as_point(self.base_point)

    @classmethod
    def euclidean_from_angles(cls, base_point: ArrayLike2D, thetas: np.ndarray) -> "GeodesicSpray2D":
        p = _as_point(base_point)
        theta_arr = np.asarray(thetas, dtype=float)
        geodesics = [Geodesic2D.from_angle(p, theta, name=fr"$\\gamma_{{{i}}}$") for i, theta in enumerate(theta_arr)]
        return cls(base_point=p, geodesics=geodesics, parameters=theta_arr)

    def front_at(self, s: float) -> np.ndarray:
        return np.array([g.point(s) for g in self.geodesics])

    def fronts_at(self, s_values: list[float] | np.ndarray) -> list[np.ndarray]:
        return [self.front_at(float(s)) for s in s_values]
