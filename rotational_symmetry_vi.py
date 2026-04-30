##############################################################################
## Rotational symmetry reverse-KL example (Section 5, Figure 2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

def coth(x): return 1.0 / np.tanh(x)
def A3(kappa): return coth(kappa) - 1.0 / kappa
def B3(kappa): return 1.0 - (3.0 / kappa) * coth(kappa) + 3.0 / kappa**2

def log_target(t, lam, eta): return lam * t - eta * t**2 # scalar profile as function of t = u^Tx
def log_vmf(X, Y, Z, nu, kappa): return kappa * (nu[0] * X + nu[1] * Y + nu[2] * Z)

def unit(v): v = np.asarray(v, dtype=float); return v / np.linalg.norm(v)
def shift_by_max(L): return L - np.nanmax(L)

# Lambert azimuthal equal-area projection centred at u = (0, 0, -1).
def lambert_forward(v):
    vx, vy, vz = unit(v)
    factor = np.sqrt(2.0 / (1.0 - vz))
    return factor * vx, factor * vy

def lambert_inverse(x, y):
    rho2 = x * x + y * y
    # only invert projection from inside the projected disk 
    mask = rho2 <= 4.0 
    X, Y, Z = (np.full_like(x, np.nan, dtype=float) for _ in range(3))

    factor = np.sqrt(1.0 - rho2[mask] / 4.0)

    X[mask] = factor * x[mask]
    Y[mask] = factor * y[mask]
    Z[mask] = rho2[mask] / 2.0 - 1.0

    return X, Y, Z, mask

def plot_sphere_example(lam=1.0, kappa0=2.0, eta_good=1.0, eta_bad=2.0, N=1001, output_path="vi_sphere_example.pdf"):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.titlepad": 4,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    u = np.array([0.0, 0.0, -1.0])
    A, B = A3(kappa0), B3(kappa0)

    # constrained minimiser for eta_bad c*, radius of latitude circle s* and selected minimiser nu_bad
    c_star = np.clip(lam * A / (2.0 * eta_bad * B), -1.0, 1.0)
    s_star = np.sqrt(max(0.0, 1.0 - c_star**2))
    nu_bad = unit([s_star, 0.0, -c_star])

    # create grid and map to sphere points inside the projected disk
    grid = np.linspace(-2.0, 2.0, N)
    XXp, YYp = np.meshgrid(grid, grid)
    Xs, Ys, Zs, mask = lambert_inverse(XXp, YYp)
    T = -Zs     # because T = u^Tx and u = (0, 0, -1)

    # compute projected log densities by pullback of the spherical density field
    LP_good = shift_by_max(np.where(mask, log_target(T, lam, eta_good), np.nan))
    LP_bad  = shift_by_max(np.where(mask, log_target(T, lam, eta_bad),  np.nan))
    LQ_good = shift_by_max(np.where(mask, log_vmf(Xs, Ys, Zs, u, kappa0), np.nan))
    LQ_bad  = shift_by_max(np.where(mask, log_vmf(Xs, Ys, Zs, nu_bad, kappa0), np.nan))

    # setup fig and panels
    fig = plt.figure(figsize=(6.75, 1.95), constrained_layout=False)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.24, 1.0, 1.0],
                          left=0.015, right=0.995, bottom=0.06, top=0.88, wspace=0.06)
    grid_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    grid_axes[2].axis("off")
    axes = [grid_axes[0], grid_axes[1], grid_axes[3], grid_axes[4]]

    panels = [
        (axes[0], LP_good, r"Target with $\eta < \eta_c$"),
        (axes[1], LQ_good, r"VI optimum with $\eta < \eta_c$"),
        (axes[2], LP_bad,  r"Target with $\eta > \eta_c$"),
        (axes[3], LQ_bad,  r"VI optimum with $\eta > \eta_c$"),
    ]

    # projection disk limits 
    t = np.linspace(0.0, 2.0 * np.pi, 600)
    bx, by = 2.0 * np.cos(t), 2.0 * np.sin(t)

    # plot log density contours 
    levels = np.linspace(-6.0, 0.0, 16)
    for ax, F, title in panels:
        ax.contourf(XXp, YYp, F, levels=levels, extend="both")
        ax.contour(XXp, YYp, F, levels=levels, linewidths=0.45, alpha=0.6)
        ax.plot(bx, by, linewidth=0.85)

        ax.set_aspect("equal", "box")
        ax.set(xlim=(-2.02, 2.02), ylim=(-2.02, 2.02), xticks=[], yticks=[], title=title)
        
        # no bounding box per sphere
        for spine in ax.spines.values(): spine.set_visible(False)

    # plot true and recovered axes + text
    for ax in axes:
        ax.plot([0.0], [0.0], marker="o", markersize=4.3, markerfacecolor="k",
                markeredgecolor="k", markeredgewidth=0.7, linestyle="None", zorder=6)

    def add_axis_marker(ax, v, label, dot_color="k"):
        px, py = lambert_forward(v)
        ax.annotate("", xy=(px, py), xytext=(0.0, 0.0), 
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.95, mutation_scale=9), zorder=7)
        ax.plot([px], [py], marker="o", markersize=3.6, markerfacecolor=dot_color,
                markeredgecolor=dot_color, markeredgewidth=0.7, linestyle="None", zorder=8)
        ax.text(px, py, f"  {label}", fontsize=8.0, va="center", zorder=9)
    
    add_axis_marker(axes[0], u, r"$u$")
    add_axis_marker(axes[1], u, r"$\nu=u$")
    add_axis_marker(axes[2], u, r"$u$")
    add_axis_marker(axes[3], nu_bad, r"$\nu$", dot_color="red")
    axes[3].text(0.05, -0.12, r"$u$", fontsize=8.5, ha="center", va="top")

    # plot circle of minimisers (possible vMF centres)
    phis = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    pts = np.array([lambert_forward([s_star * np.cos(ph), s_star * np.sin(ph), -c_star]) for ph in phis]) # nu_bad is just the point at ph = 0 
    axes[3].plot(pts[:, 0], pts[:, 1], marker="o", linestyle="None", 
                 markersize=2.1, markeredgewidth=0.55, alpha=0.9, zorder=6)

    # add divider line
    left_group_right, right_group_left = axes[1].get_position().x1, axes[2].get_position().x0
    divider_x = 0.5 * (left_group_right + right_group_left)
    axes_y0, axes_y1 = min(ax.get_position().y0 for ax in axes), max(ax.get_position().y1 for ax in axes)
    divider_y0, divider_y1 = axes_y0 + 0.03, axes_y1 - 0.02
    fig.add_artist(Line2D([divider_x, divider_x], [divider_y0, divider_y1],
                          transform=fig.transFigure, color="0.25", linewidth=1.15,
                          alpha=0.95, solid_capstyle="round", zorder=20))

    # save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved PDF to {output_path.resolve()}")

    plt.close(fig)

plot_sphere_example(lam=1.0, kappa0=2.5, eta_good=1.0, eta_bad=2.0, output_path="outputs/vi_sphere_example.pdf")
