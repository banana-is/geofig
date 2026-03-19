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


def sample_ellipse(a: float, b: float, center: tuple[float, float], n: int = 800) -> np.ndarray:
    return Ellipse(a=a, b=b, center=center).sample(n)


def main() -> None:
    # ---------------------------------
    # parameters
    # ---------------------------------
    a = 2.0
    b = 1.0

    t = 1.0          # current time
    r = 0.45         # extra time step

    # fixed translation of the entire front
    p0 = np.array([1.1, 0.6])

    n_curve = 1200
    n_wavelet = 300
    num_wavelets = 36

    # ---------------------------------
    # front at time t:
    # T_t = p0 + tE
    # ---------------------------------
    center_t = p0
    front_t = sample_ellipse(
        a=t * a,
        b=t * b,
        center=(center_t[0], center_t[1]),
        n=n_curve,
    )

    # ---------------------------------
    # front at time t+r:
    # T_{t+r} = p0 + (t+r)E
    # ---------------------------------
    center_tr = p0
    front_tr = sample_ellipse(
        a=(t + r) * a,
        b=(t + r) * b,
        center=(center_tr[0], center_tr[1]),
        n=n_curve,
    )

    # ---------------------------------
    # centered elliptic wavelet:
    # rE
    # so each wavelet at x on T_t is x + rE
    # ---------------------------------
    wavelet = EllipseIndicatrix(a=a, b=b)

    fig = Figure2D(figsize=(9, 7))

    # current front
    fig.add_polyline(
        front_t,
        closed=True,
        style=LineStyle(
            color=indigo,
            linewidth=3,
            label=r"Current front $T_t = p_0 + tE$",
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
            label=r"Next front $T_{t+r} = p_0 + (t+r)E$",
        ),
    )

    # wavelets placed along current front
    idx = np.linspace(0, n_curve - 1, num_wavelets, dtype=int)
    for k in idx:
        x = front_t[k]
        W = wavelet.sample(n_wavelet, scale=r) + x
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=deepblue, linewidth=1, alpha=0.22),
        )

    # mark the fixed center
    fig.ax.scatter(
        [p0[0]],
        [p0[1]],
        s=35,
        color=rose,
        zorder=5,
    )

    fig.ax.text(
        p0[0] + 0.05,
        p0[1] - 0.15,
        r"$p_0$",
        color=rose,
        fontsize=11,
    )

    fig.set_equal()
    fig.hide_axes()

    pad_x = abs(p0[0]) + (t + r) * a * 1.6
    pad_y = abs(p0[1]) + (t + r) * b * 1.8

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
    fig.save("outputs/figures/translated_wavelets.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()