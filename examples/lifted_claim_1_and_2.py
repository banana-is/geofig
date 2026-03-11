import numpy as np

from geofig import Ellipse, EllipseIndicatrix
from geofig import MongePatch
from geofig import Figure3D, LineStyle3D


# ---------------------------------
# palette
# ---------------------------------
gold     = "#F8B55F"
rose     = "#C95792"
violet   = "#7C4585"
indigo   = "#3D365C"
deepblue = "#2E365C"


def euclidean_offset_of_ellipse(a: float, b: float, r: float, n: int = 2000) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    x = a * np.cos(t)
    y = b * np.sin(t)

    gx = x / (a * a)
    gy = y / (b * b)
    gn = np.sqrt(gx**2 + gy**2)

    nx = gx / gn
    ny = gy / gn

    return np.column_stack([x + r * nx, y + r * ny])


def main() -> None:
    # ---------------------------------
    # parameters
    # ---------------------------------
    a = 2.0
    b = 1.0
    r = 0.35
    n_curve = 1600

    num_circle_wavelets = 30
    num_ellipse_wavelets = 30
    n_wavelet = 220

    # ---------------------------------
    # planar curves
    # ---------------------------------
    ellipse = Ellipse(a=a, b=b)
    front_pts = ellipse.sample(n_curve)

    euc_env = euclidean_offset_of_ellipse(a=a, b=b, r=r, n=n_curve)
    ellipse_env = Ellipse(a=(1 + r) * a, b=(1 + r) * b).sample(n_curve)

    circle_wavelet = EllipseIndicatrix(a=1.0, b=1.0)
    ellipse_wavelet = EllipseIndicatrix(a=a, b=b)

    # ---------------------------------
    # surface
    # ---------------------------------
    surface = MongePatch.mountain()

    pad = (1 + r) * a * 1.4
    xlim = (-pad, pad)
    ylim = (-pad, pad)

    X, Y, Z = surface.grid(xlim=xlim, ylim=ylim, nx=220, ny=220)

    # ---------------------------------
    # figure
    # ---------------------------------
    fig = Figure3D(figsize=(12, 8))
    fig.add_surface(X, Y, Z, cmap="Greys", alpha=0.30)

    # lifted main curves
    fig.add_polyline(
        surface.lift_curve(front_pts),
        closed=True,
        style=LineStyle3D(
            color=indigo,
            linewidth=3.6,
            label="Initial ellipse",
        ),
    )

    fig.add_polyline(
        surface.lift_curve(euc_env),
        closed=True,
        style=LineStyle3D(
            color=rose,
            linewidth=3.0,
            label="Euclidean Huygens envelope - offset by circles - not an ellipse",
        ),
    )

    fig.add_polyline(
        surface.lift_curve(ellipse_env),
        closed=True,
        style=LineStyle3D(
            color=gold,
            linewidth=3.0,
            linestyle="--",
            label="Elliptic-metric Huygens envelope - offset by ellipses - homothety",
        ),
    )

    # circle wavelets
    idx_c = np.linspace(0, n_curve - 1, num_circle_wavelets, dtype=int)
    for k in idx_c:
        p = front_pts[k]
        W = circle_wavelet.sample(n_wavelet, scale=r) + p
        fig.add_polyline(
            surface.lift_curve(W),
            closed=True,
            style=LineStyle3D(color=violet, linewidth=1.1, alpha=0.16),
        )

    # ellipse wavelets
    idx_e = np.linspace(0, n_curve - 1, num_ellipse_wavelets, dtype=int)
    for k in idx_e:
        p = front_pts[k]
        W = ellipse_wavelet.sample(n_wavelet, scale=r) + p
        fig.add_polyline(
            surface.lift_curve(W),
            closed=True,
            style=LineStyle3D(color=deepblue, linewidth=1.1, alpha=0.14),
        )

    fig.set_view(elev=34, azim=-58)
    fig.set_limits(xlim=xlim, ylim=ylim)
    fig.hide_axes()

    fig.add_legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()
    fig.save("outputs/figures/lifted_elliptic_wavelets.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()