"""Example scene for Exercise 4.2.
"""

from pathlib import Path

import numpy as np

from geofig import GeodesicSpray2D
from geofig import GeodesicSprayScene


def main() -> None:
    p = np.array([1.0, 0.0])
    thetas = np.linspace(-np.pi, np.pi, 17, endpoint=False)

    spray = GeodesicSpray2D.euclidean_from_angles(base_point=p, thetas=thetas)
    scene = GeodesicSprayScene(
        spray=spray,
        s_min=-2.2,
        s_max=2.2,
        n_samples=300,
        front_radii=[0.75, 1.5, 2.25],
        highlight_index=10,
        title="Exercise 4.2: Euclidean geodesic spray from p=(1,0)",
    )

    out = Path("outputs/figures/exercise_4_2_scene.png")
    scene.save(out)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
