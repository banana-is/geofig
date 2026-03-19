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


def euclidean_offset_of_ellipse(a: float, b: float, r: float, center=(0.0, 0.0), n: int = 2000) -> np.ndarray:
    """
    Euclidean parallel curve to the ellipse
        x = center + (a cos t, b sin t)
    obtained by moving distance r along the outward Euclidean unit normal.
    """
    cx, cy = center
    t = np.linspace(0, 2 * np.pi, n)

    x = a * np.cos(t)
    y = b * np.sin(t)

    gx = x / (a * a)
    gy = y / (b * b)
    gn = np.sqrt(gx**2 + gy**2)

    nx = gx / gn
    ny = gy / gn

    return np.column_stack([cx + x + r * nx, cy + y + r * ny])


def sample_ellipse(a: float, b: float, center=(0.0, 0.0), n: int = 2000) -> np.ndarray:
    return Ellipse(a=a, b=b, center=center).sample(n)


def shifted_sample(indicatrix: EllipseIndicatrix, scale: float, shift=(0.0, 0.0), n: int = 400) -> np.ndarray:
    sx, sy = shift
    return indicatrix.sample(n, scale=scale) + np.array([sx, sy])


def main() -> None:
    # ---------------------------------
    # common parameters
    # ---------------------------------
    a = 2.0
    b = 1.0
    t = 1.0
    r = 0.35

    n_curve = 2000
    n_wavelet = 400
    num_wavelets = 42

    # ---------------------------------
    # placement of the three scenes
    # ---------------------------------
    center_12 = np.array([0.0, 0.0])       # claims 1 and 2 in the middle
    p3 = np.array([-6.4, -4.0])            # claim 3 somewhere else
    p4 = np.array([6.4, -4.2])             # claim 4 somewhere else

    # drift vector for claim 4
    c = np.array([0.9, 0.35])

    # ---------------------------------
    # figure
    # ---------------------------------
    fig = Figure2D(figsize=(12, 10))

    # ============================================================
    # CLAIMS 1 and 2 together at the center
    # ============================================================
    front_12 = sample_ellipse(a=a, b=b, center=center_12, n=n_curve)

    # claim 1 envelope
    euc_env = euclidean_offset_of_ellipse(
        a=a, b=b, r=r, center=center_12, n=n_curve
    )

    # claim 2 envelope
    ell_env = sample_ellipse(
        a=(1 + r) * a,
        b=(1 + r) * b,
        center=center_12,
        n=n_curve,
    )

    # initial ellipse
    fig.add_polyline(
        front_12,
        closed=True,
        style=LineStyle(
            color=indigo,
            linewidth=3,
            label="Initial ellipse",
        ),
    )

    # claim 1: Euclidean envelope
    fig.add_polyline(
        euc_env,
        closed=True,
        style=LineStyle(
            color=rose,
            linewidth=3,
            label="Claim 1: Euclidean Huygens envelope (circle wavelets) — not an ellipse",
        ),
    )

    # claim 2: elliptic envelope
    fig.add_polyline(
        ell_env,
        closed=True,
        style=LineStyle(
            color=gold,
            linewidth=3,
            linestyle="--",
            label="Claim 2: elliptic-metric envelope (elliptic wavelets) — homothetic ellipse",
        ),
    )

    # claim 1 wavelets: circles
    circle_wavelet = EllipseIndicatrix(a=1.0, b=1.0)
    idx = np.linspace(0, n_curve - 1, num_wavelets, dtype=int)
    for k in idx:
        p = front_12[k]
        W = circle_wavelet.sample(n_wavelet, scale=r) + p
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=violet, linewidth=1, alpha=0.18),
        )

    # claim 2 wavelets: ellipses
    ellipse_wavelet = EllipseIndicatrix(a=a, b=b)
    for k in idx:
        p = front_12[k]
        W = ellipse_wavelet.sample(n_wavelet, scale=r) + p
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=deepblue, linewidth=1, alpha=0.18),
        )

    fig.ax.text(
        center_12[0] - 1.5,
        center_12[1] + 2.45,
        "Claims 1 & 2",
        color=indigo,
        fontsize=14,
        fontweight="bold",
    )

    # ============================================================
    # CLAIM 3: translated elliptic picture
    # T_t = p3 + tE,  T_{t+r} = p3 + (t+r)E
    # ============================================================
    center3_t = p3
    center3_tr = p3

    front3_t = sample_ellipse(
        a=t * a,
        b=t * b,
        center=center3_t,
        n=n_curve,
    )
    front3_tr = sample_ellipse(
        a=(t + r) * a,
        b=(t + r) * b,
        center=center3_tr,
        n=n_curve,
    )

    fig.add_polyline(
        front3_t,
        closed=True,
        style=LineStyle(
            color=indigo,
            linewidth=3,
            label=r"Claim 3: translated front $T_t=p_0+tE$",
        ),
    )

    fig.add_polyline(
        front3_tr,
        closed=True,
        style=LineStyle(
            color=gold,
            linewidth=3,
            linestyle="--",
            label=r"Claim 3: evolved front $T_{t+r}=p_0+(t+r)E$",
        ),
    )

    for k in idx:
        p = front3_t[k]
        W = ellipse_wavelet.sample(n_wavelet, scale=r) + p
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=deepblue, linewidth=1, alpha=0.16),
        )

    fig.ax.scatter(
        [p3[0]],
        [p3[1]],
        s=36,
        color=rose,
        zorder=5,
    )
    fig.ax.text(
        p3[0] + 0.10,
        p3[1] - 0.20,
        r"$p_0$",
        color=rose,
        fontsize=12,
    )
    fig.ax.text(
        p3[0] - 1.4,
        p3[1] + 2.35,
        "Claim 3",
        color=indigo,
        fontsize=14,
        fontweight="bold",
    )

    # ============================================================
    # CLAIM 4: translated + shifted
    # T_t = p4 + tc + tE,  T_{t+r} = p4 + (t+r)c + (t+r)E
    # with fixed ignition point p4
    # ============================================================
    center4_t = p4 + t * c
    center4_tr = p4 + (t + r) * c
    rc = r * c

    front4_t = sample_ellipse(
        a=t * a,
        b=t * b,
        center=center4_t,
        n=n_curve,
    )
    front4_tr = sample_ellipse(
        a=(t + r) * a,
        b=(t + r) * b,
        center=center4_tr,
        n=n_curve,
    )

    fig.add_polyline(
        front4_t,
        closed=True,
        style=LineStyle(
            color=indigo,
            linewidth=3,
            label=r"Claim 4: drifting front $T_t=p_0+tc+tE$",
        ),
    )

    fig.add_polyline(
        front4_tr,
        closed=True,
        style=LineStyle(
            color=gold,
            linewidth=3,
            linestyle="--",
            label=r"Claim 4: evolved front $T_{t+r}=p_0+(t+r)c+(t+r)E$",
        ),
    )

    for k in idx:
        p = front4_t[k]
        W = ellipse_wavelet.sample(n_wavelet, scale=r) + p + rc
        fig.add_polyline(
            W,
            closed=True,
            style=LineStyle(color=deepblue, linewidth=1, alpha=0.16),
        )

    fig.ax.scatter(
        [p4[0], center4_t[0], center4_tr[0]],
        [p4[1], center4_t[1], center4_tr[1]],
        s=36,
        color=rose,
        zorder=5,
    )

    fig.ax.annotate(
        "",
        xy=center4_tr,
        xytext=center4_t,
        arrowprops=dict(arrowstyle="->", color=rose, lw=2),
    )

    fig.ax.text(
        p4[0] + 0.10,
        p4[1] - 0.20,
        r"$p_0$",
        color=rose,
        fontsize=12,
    )
    fig.ax.text(
        center4_t[0] + 0.10,
        center4_t[1] - 0.20,
        r"$p_0+tc$",
        color=rose,
        fontsize=12,
    )
    fig.ax.text(
        center4_tr[0] + 0.10,
        center4_tr[1] - 0.20,
        r"$p_0+(t+r)c$",
        color=rose,
        fontsize=12,
    )
    fig.ax.text(
        p4[0] - 1.2,
        p4[1] + 2.65,
        "Claim 4",
        color=indigo,
        fontsize=14,
        fontweight="bold",
    )

    # ---------------------------------
    # final formatting
    # ---------------------------------
    fig.set_equal()
    fig.hide_axes()

    fig.set_limits(-10.5, 10.5, -8.0, 4.5)

    fig.add_legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()
    fig.save("outputs/figures/combined_wavelets.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()