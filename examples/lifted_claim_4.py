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


def main() -> None:

    # ---------------------------------
    # parameters
    # ---------------------------------
    a = 2.0
    b = 1.0

    t = 1.0
    r = 0.45

    # drift / wind vector
    c = np.array([0.9, 0.35])

    n_curve = 1200
    n_wavelet = 220
    num_wavelets = 28

    # ---------------------------------
    # fronts in the plane
    # ---------------------------------
    center_t = t * c
    center_tr = (t + r) * c

    front_t = Ellipse(
        a=t * a,
        b=t * b,
        center=(center_t[0], center_t[1]),
    ).sample(n_curve)

    front_tr = Ellipse(
        a=(t + r) * a,
        b=(t + r) * b,
        center=(center_tr[0], center_tr[1]),
    ).sample(n_curve)

    # wavelet prototype
    wavelet = EllipseIndicatrix(a=a, b=b)

    rc = r * c

    # ---------------------------------
    # surface
    # ---------------------------------
    surface = MongePatch.mountain()

    pad = (t + r) * a * 1.5 + np.linalg.norm(center_tr)

    xlim = (-pad, pad)
    ylim = (-pad, pad)

    X, Y, Z = surface.grid(xlim=xlim, ylim=ylim, nx=220, ny=220)

    # ---------------------------------
    # figure
    # ---------------------------------
    fig = Figure3D(figsize=(12, 8))

    fig.add_surface(
        X, Y, Z,
        cmap="Greys",
        alpha=0.30,
    )

    # lifted fronts
    fig.add_polyline(
        surface.lift_curve(front_t),
        closed=True,
        style=LineStyle3D(
            color=indigo,
            linewidth=3.6,
            label=r"Current front $T_t = tc + tE$",
        ),
    )

    fig.add_polyline(
        surface.lift_curve(front_tr),
        closed=True,
        style=LineStyle3D(
            color=gold,
            linewidth=3.0,
            linestyle="--",
            label=r"Next front $T_{t+r} = (t+r)c + (t+r)E$",
        ),
    )

    # wavelets
    idx = np.linspace(0, n_curve - 1, num_wavelets, dtype=int)

    for k in idx:
        p = front_t[k]
        W = wavelet.sample(n_wavelet, scale=r) + p + rc

        fig.add_polyline(
            surface.lift_curve(W),
            closed=True,
            style=LineStyle3D(
                color=deepblue,
                linewidth=1.2,
                alpha=0.18,
            ),
        )

    # ---------------------------------
    # drift arrow
    # ---------------------------------
    centers = np.array([center_t, center_tr])

    lifted_centers = surface.lift_curve(centers)

    fig.add_points(
        lifted_centers,
        color=rose,
        s=40,
    )

    # draw drift segment
    fig.add_polyline(
        lifted_centers,
        style=LineStyle3D(
            color=rose,
            linewidth=2.4,
        ),
    )

    # ---------------------------------
    # styling
    # ---------------------------------
    fig.set_view(elev=34, azim=-55)

    fig.set_limits(
        xlim=xlim,
        ylim=ylim,
    )

    fig.hide_axes()

    fig.add_legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()

    fig.save(
        "outputs/figures/lifted_shifted_wavelets.png",
        dpi=300,
    )

    fig.show()


if __name__ == "__main__":
    main()