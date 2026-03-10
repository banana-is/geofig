import numpy as np

from geofig import Ellipse, EllipseIndicatrix, Figure2D, LineStyle

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
    a = 2.0
    b = 1.0
    r = 0.35
    n_curve = 2000

    num_circle_wavelets = 50
    num_ellipse_wavelets = 50

    ellipse = Ellipse(a=a, b=b)
    front_pts = ellipse.sample(n_curve)

    # claim 1: euclidean wavelets -> euclidean offset
    euc_env = euclidean_offset_of_ellipse(a=a, b=b, r=r, n=n_curve)

    # claim 2: elliptic-metric wavelets -> homothety
    ellipse_env = Ellipse(a=(1 + r) * a, b=(1 + r) * b).sample(n_curve)

    fig = Figure2D(figsize=(8, 8))

    # initial ellipse
    fig.add_polyline(
        front_pts,
        closed=True,
        style=LineStyle(color=indigo, linewidth=3, label="Initial ellipse"),
    )

    # euclidean envelope
    fig.add_polyline(
        euc_env,
        closed=True,
        style=LineStyle(
            color=rose,
            linewidth=3,
            label="Euclidean Huygens envelope - offset by circles - not an ellipse",
        ),
    )

    # elliptic envelope
    fig.add_polyline(
        ellipse_env,
        closed=True,
        style=LineStyle(
            color=gold,
            linewidth=3,
            linestyle="--",
            label="Elliptic-metric Huygens envelope - offset by ellipses - homothety",
        ),
    )

    # circle wavelets
    circle_wavelet = EllipseIndicatrix(a=1.0, b=1.0)
    idx_c = np.linspace(0, n_curve - 1, num_circle_wavelets, dtype=int)
    for k in idx_c:
        p = front_pts[k]
        W = circle_wavelet.sample(400, scale=r) + p
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=violet, linewidth=1, alpha=0.25),
        )

    # ellipse wavelets
    ellipse_wavelet = EllipseIndicatrix(a=a, b=b)
    idx_e = np.linspace(0, n_curve - 1, num_ellipse_wavelets, dtype=int)
    for k in idx_e:
        p = front_pts[k]
        W = ellipse_wavelet.sample(400, scale=r) + p
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=deepblue, linewidth=1, alpha=0.25),
        )

    fig.set_equal()
    fig.hide_axes()

    pad = (1 + r) * a * 1.2
    fig.set_limits(-pad, pad, -pad, pad)

    fig.add_legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()
    fig.save("outputs/figures/ellipse_wavelets.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()