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


def sample_shifted_ellipse(a: float, b: float, center: tuple[float, float], n: int = 800) -> np.ndarray:
    return Ellipse(a=a, b=b, center=center).sample(n)


def main() -> None:
    # ---------------------------------
    # parameters
    # ---------------------------------
    a = 2.0
    b = 1.0

    t = 1.0          # current time
    r = 0.45         # extra time step

    # drift / wind shift vector
    c = np.array([0.9, 0.35])

    n_curve = 1200
    n_wavelet = 300
    num_wavelets = 36

    # ---------------------------------
    # front at time t:
    # T_t = t(c + E) = tc + tE
    # ---------------------------------
    center_t = t * c
    front_t = sample_shifted_ellipse(
        a=t * a,
        b=t * b,
        center=(center_t[0], center_t[1]),
        n=n_curve,
    )

    # ---------------------------------
    # front at time t+r:
    # T_{t+r} = (t+r)(c + E)
    # ---------------------------------
    center_tr = (t + r) * c
    front_tr = sample_shifted_ellipse(
        a=(t + r) * a,
        b=(t + r) * b,
        center=(center_tr[0], center_tr[1]),
        n=n_curve,
    )

    # ---------------------------------
    # shifted wavelet:
    # rK = r(c + E) = rc + rE
    # so each wavelet centered at x on T_t is x + rc + rE
    # ---------------------------------
    rc = r * c
    wavelet = EllipseIndicatrix(a=a, b=b)

    fig = Figure2D(figsize=(9, 7))

    # current front
    fig.add_polyline(
        front_t,
        closed=True,
        style=LineStyle(
            color=indigo,
            linewidth=3,
            label=r"Current front $T_t = tc + tE$",
        ),
    )

    # evolved front
    fig.add_polyline(
        front_tr,
        closed=True,
        style=LineStyle(
            color=gold,
            linewidth=3,
            linestyle="--",
            label=r"Next front $T_{t+r} = (t+r)c + (t+r)E$",
        ),
    )

    # wavelets placed along current front
    idx = np.linspace(0, n_curve - 1, num_wavelets, dtype=int)
    for k in idx:
        p = front_t[k]
        W = wavelet.sample(n_wavelet, scale=r) + p + rc
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=deepblue, linewidth=1, alpha=0.22),
        )

    # mark centers
    # current center tc
    # next center (t+r)c
    centers = np.array([
        center_t,
        center_tr,
    ])
    fig.ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=35,
        color=rose,
        zorder=5,
    )

    # optional drift arrow
    fig.ax.annotate(
        "",
        xy=center_tr,
        xytext=center_t,
        arrowprops=dict(arrowstyle="->", color=rose, lw=2),
    )

    fig.ax.text(
        center_t[0] + 0.05,
        center_t[1] - 0.15,
        r"$tc$",
        color=rose,
        fontsize=11,
    )
    fig.ax.text(
        center_tr[0] + 0.05,
        center_tr[1] - 0.15,
        r"$(t+r)c$",
        color=rose,
        fontsize=11,
    )

    fig.set_equal()
    fig.hide_axes()

    pad_x = (t + r) * a * 1.6 + abs(center_tr[0])
    pad_y = (t + r) * b * 1.8 + abs(center_tr[1])

    fig.set_limits(
        -pad_x, pad_x,
        -pad_y, pad_y,
    )

    fig.add_legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()
    fig.save("outputs/figures/shifted_wavelets.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()