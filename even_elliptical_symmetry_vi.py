import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

##############################################################################
## Even symmetry reverse-KL example (Section 4, Figure 1, Left panel)

lam = 9
v_coeffs = {2: 1.1, 4: -2.9, 6: 3.0, 8: -1.3, 10: 0.2}
max_moment_deg = max(v_coeffs)

def V(z): return sum(coeff * z**deg for deg, coeff in v_coeffs.items())
def unnorm_p(x): return np.exp(-lam * V(x))

def p(x):
    Z, _ = quad(unnorm_p, -np.inf, np.inf, limit=300)
    return unnorm_p(x) / Z

def gaussian_raw_moments(nu, s, max_deg=max_moment_deg):
    """
    Recurrence:
        M_0 = 1,
        M_1 = nu,
        M_n = nu M_{n-1} + (n - 1) s^2 M_{n-2},   n >= 2.
    """
    moments = np.empty(max_deg + 1)
    moments[0], moments[1] = 1.0, nu
    
    for n in range(2, max_deg + 1):
        moments[n] = nu * moments[n - 1] + (n - 1) * s * s * moments[n - 2]

    return moments

def objective_and_grad(x):
    nu, s = x
    moments = gaussian_raw_moments(nu, s)

    # L(nu,s) = -log(s) + lam * E[V(X)],  X ~ N(nu, s^2)
    val = -math.log(s) + lam * sum(coeff * moments[deg] for deg, coeff in v_coeffs.items())

    # d/dnu M_n = n M_{n-1}
    dL_dnu = lam * sum(coeff * deg * moments[deg - 1] for deg, coeff in v_coeffs.items())
    # d/ds M_n = n(n-1)s M_{n-2}
    dL_ds = -1.0 / s + lam * sum(coeff * deg * (deg - 1) * s * moments[deg - 2] for deg, coeff in v_coeffs.items())

    return val, np.array([dL_dnu, dL_ds])

def compute_left_panel_data():
    nus = [-1.75, -1.25, -0.75, 0.75, 1.25, 1.75]
    ss = [0.2, 0.5, 0.9]
    starts = [(nu, s) for nu in nus for s in ss]

    results = []
    for start in starts:
        res = minimize(fun=objective_and_grad, x0=np.array(start), jac=True, method="L-BFGS-B", bounds=[(-3.0, 3.0), (0.05, 2.0)], options={"maxiter": 500})
        results.append(res)

    best = min(results, key=lambda res: res.fun)
    nu_star, s_star = best.x

    x = np.linspace(-3, 3, 2000)
    p_vals = p(x)
    q_vals = norm.pdf(x, loc=nu_star, scale=s_star)

    return {"x": x, "p_vals": p_vals, "q_vals": q_vals, "nu_star": nu_star, "s_star": s_star, "mean": 0.0}

##############################################################################
## Elliptical symmetry reverse-KL example (Section 4, Figure 1, Right panel)

def sqrtm_spd(M):
    evals, evecs = np.linalg.eigh(M)
    return (evecs * np.sqrt(evals)) @ evecs.T

# If p(y) \propto exp(-||y||^{2k}) on R^2, then Cov(Y) = tau_k I_2
def tau_k(k): return 0.5 * math.gamma(2.0 / k) / math.gamma(1.0 / k)

def sample_target_2d(n, mu, Sigma, k, rng=None):
    """
    Sample from p(x) \propto exp(-(((x-mu)^T Sigma^{-1} (x-mu))^k)) as: 
    - s ~ Gamma(1/k, 1), r = s^{1/2k}
    - theta ~ Unif[0, 2pi)
    - y = r (cos theta, sin theta)^T 
    - x = mu + Sigma^{1/2}y
    """
    rng = np.random.default_rng(rng)

    s = rng.gamma(shape=1.0 / k, scale=1.0, size=n) 
    r = s ** (1.0 / (2.0 * k))
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    Y = r[:, None] * directions

    return mu + Y @ sqrtm_spd(Sigma).T

def ellipse_points(C, m, r=2.0, n=400):
    # draw points on ellipse E_r = m + r*C^{1/2}*S^1 
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    return m + r * (circle @ sqrtm_spd(C).T)

def target_y1_marginal(z, k):
    c_k = k / (math.pi * math.gamma(1.0 / k))
    integral, _ = quad(lambda t: math.exp(-((z * z + t * t) ** k)), 0.0, np.inf, limit=200)
    return 2.0 * c_k * integral

def target_x1_marginal(x1, mu, Sigma, k):
    scale = math.sqrt(Sigma[0, 0])
    z = (x1 - mu[0]) / scale
    return target_y1_marginal(z, k) / scale

k = 15
bin_coeffs = np.array([math.comb(k, j) for j in range(k + 1)])

def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def right_objective_and_grad(params):
    """
    minimise KL[q_Y || p_Y] over Gaussian q_Y. 
    Let Z = R(theta)^T Y. Then Z \sim N(u, diag(t1^2, t2^2)) where u = R(theta)^T a, 
    and up to constants, L = -log t1 - log t2 + E[(Z1^2 + Z2^2)^K].
    """
    a = np.array(params[:2])
    theta, t1, t2 = params[2:]

    R = rot2(theta)
    u1, u2 = R.T @ a

    # M_i[n] = E[Z_i^n].
    M1 = gaussian_raw_moments(u1, t1, max_deg=2 * k)
    M2 = gaussian_raw_moments(u2, t2, max_deg=2 * k)

    L = -math.log(t1) - math.log(t2)

    dL_du1, dL_du2 = 0.0, 0.0
    dL_dt1, dL_dt2 = -1.0 / t1, -1.0 / t2

    for j, coeff in enumerate(bin_coeffs):
        n1 = 2 * j
        n2 = 2 * (k - j)

        # E[(Z1^2 + Z2^2)^K]
        L += coeff * M1[n1] * M2[n2]

        #   d/du E[Z^n] = n E[Z^{n-1}]
        #   d/dt E[Z^n] = n(n-1)t E[Z^{n-2}]
        if n1 > 0:
            dL_du1 += coeff * n1 * M1[n1 - 1] * M2[n2]
            dL_dt1 += coeff * n1 * (n1 - 1) * t1 * M1[n1 - 2] * M2[n2]

        if n2 > 0:
            dL_du2 += coeff * M1[n1] * n2 * M2[n2 - 1]
            dL_dt2 += coeff * M1[n1] * n2 * (n2 - 1) * t2 * M2[n2 - 2]

    # Chain rule: u = R(theta)^T a.
    dL_du = np.array([dL_du1, dL_du2])
    dL_da = R @ dL_du       
    dL_dtheta = dL_du1 * u2 - dL_du2 * u1

    grad = np.array([dL_da[0], dL_da[1], dL_dtheta, dL_dt1, dL_dt2])
    return L, grad

def fit_right_panel_vi(mu, Sigma):
    """
    To find a minimiser of KL[q_X || p_X] over Gaussian q_X, find q_Y^* minimiser of KL[q_Y || p_Y] 
    over Gaussian q_Y and push forward q_X^* = T_# q_Y^* where T(y) = mu + Sigma^{1/2}y. 
    """
    A = sqrtm_spd(Sigma)
    
    # Parameterise Gaussian Q_Y = N(a, C) where a = (a1, a2), and 
    # C = R(theta) diag(t1^2, t2^2) R(theta)^T, for t1, t2 > 0.
    starts = [np.array([a1, a2, theta, t1, t2])
        for a1 in (-1.0, -0.5, 0.5, 1.0)
        for a2 in (-1.0, -0.5, 0.5, 1.0)
        for theta in (-1.0, 0.0, 1.0)
        for t1 in (0.2, 0.5, 0.9)
        for t2 in (0.2, 0.5, 0.9)]

    bounds = [
        (-2.0, 2.0),          # a1
        (-2.0, 2.0),          # a2
        (-math.pi, math.pi),  # theta
        (0.05, 2.0),          # t1
        (0.05, 2.0),          # t2
    ]

    results = [minimize(fun=right_objective_and_grad, x0=x0, jac=True, method="L-BFGS-B", 
                        bounds=bounds, options={"maxiter": 500}) for x0 in starts]
    best = min(results, key=lambda res: res.fun)

    a_star = best.x[:2]
    theta_star, t1_star, t2_star = best.x[2:]

    # reconstruct Q_Y^* from optimal parameters
    R = rot2(theta_star)
    C_star = R @ np.diag([t1_star**2, t2_star**2]) @ R.T

    # Map Y ~ N(a_star, C_star) back to X = mu + Sigma^{1/2} Y.
    nu_star = mu + A @ a_star
    cov_q = A @ C_star @ A.T

    return {"nu_star": nu_star, "cov_q": cov_q}

def compute_right_panel_data():
    mu2 = np.array([2.0, -1.0])
    Sigma2 = np.array([[4.0, 3.0], [3.0, 9.0]])

    tau = tau_k(k)
    cov_p = tau * Sigma2
    samples_p = sample_target_2d(3000, mu2, Sigma2, k, rng=0)
    ep = ellipse_points(cov_p, m=mu2, r=2.0)

    fit = fit_right_panel_vi(mu2, Sigma2)
    nu_star, cov_q = fit["nu_star"], fit["cov_q"]

    rng = np.random.default_rng(1)
    samples_q = rng.multivariate_normal(nu_star, cov_q, size=3000)
    eq = ellipse_points(cov_q, m=nu_star, r=2.0)

    # find limits for plotting
    all_pts = np.vstack([samples_p, samples_q, ep, eq, mu2.reshape(1, 2), nu_star.reshape(1, 2)])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    spans = maxs - mins
    xlim = (mins[0] - 0.10 * spans[0], maxs[0] + 0.25 * spans[0])
    ylim = (mins[1] - 0.10 * spans[1], maxs[1] + 0.25 * spans[1])

    # Exact target X_1 marginal versus fitted Gaussian X_1 marginal
    x1_grid = np.linspace(xlim[0], xlim[1], 401)
    target_x1 = np.array([target_x1_marginal(x, mu2, Sigma2, k) for x in x1_grid])
    gauss_x1 = norm.pdf(x1_grid, loc=nu_star[0], scale=np.sqrt(cov_q[0, 0]))

    marginal_xlim = (x1_grid.min(), x1_grid.max())
    marginal_ylim = (0.0, 1.08 * max(target_x1.max(), gauss_x1.max()))
    
    return {
        "samples_p": samples_p,
        "samples_q": samples_q,
        "ep": ep,
        "eq": eq,
        "mu2": mu2,
        "x1_grid": x1_grid,
        "target_x1": target_x1,
        "gauss_x1": gauss_x1,
        "marginal_xlim": marginal_xlim,
        "marginal_ylim": marginal_ylim,
        "xlim": xlim,
        "ylim": ylim
    }
    
################################# 
## Plotting two panels together
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"

def plot_figure(left_data, right_data):
    fig_width = 12.4
    fig_height = 0.24 * fig_width

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={"width_ratios": [1, 1]}, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.22, top=0.96, wspace=0.24)

    # Left panel
    ax = axes[0]

    line_p, = ax.plot(left_data["x"], left_data["p_vals"], color=BLUE, lw=2.2, label=r"$p$")
    line_q, = ax.plot(left_data["x"], left_data["q_vals"], color=ORANGE, linewidth=2.2, linestyle="--", label=r"$q$")
    line_m = ax.axvline(left_data["mean"], color="0.25", lw=1.25, ls="-.", label="mean")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    ax.set_xlim(left_data["x"].min(), left_data["x"].max())
    ax.set_ylim(bottom=0)

    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.legend(handles=[line_p, line_q, line_m], loc="upper right", frameon=False, handlelength=3.0)

    # Right panel
    ax = axes[1]

    ax.scatter(right_data["samples_p"][:, 0], right_data["samples_p"][:, 1], s=4, alpha=0.14, color=BLUE)
    ax.scatter(right_data["samples_q"][:, 0], right_data["samples_q"][:, 1], s=4, alpha=0.10, color=ORANGE)

    line_p2, = ax.plot(right_data["ep"][:, 0], right_data["ep"][:, 1], color=BLUE, linewidth=2.2, label=r"$p$")
    line_q2, = ax.plot(right_data["eq"][:, 0], right_data["eq"][:, 1], color=ORANGE, linestyle="--", linewidth=2.2, label=r"$q$")

    mean_pt = ax.scatter(*right_data["mu2"], marker="x", s=90, linewidths=2, color="0.25", label="mean", zorder=5)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(*right_data["xlim"])
    ax.set_ylim(*right_data["ylim"])

    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.legend(handles=[line_p2, line_q2, mean_pt], loc="upper right", bbox_to_anchor=(0.985, 0.985), borderaxespad=0.0, frameon=False, handlelength=3.0)

    # inset
    axins = ax.inset_axes([0.02, 0.78, 0.32, 0.22], facecolor="white", zorder=6)

    axins.plot(right_data["x1_grid"], right_data["target_x1"], color=BLUE, lw=1.6)
    axins.plot(right_data["x1_grid"], right_data["gauss_x1"], color=ORANGE, lw=1.6, ls=(0, (6, 2)))

    axins.set_xlim(*right_data["marginal_xlim"])
    axins.set_ylim(*right_data["marginal_ylim"])
    axins.locator_params(axis="x", nbins=3)
    axins.set_yticks([])
    axins.tick_params(labelsize=7, pad=1.2, length=2.2)
    axins.ticklabel_format(axis="x", style="plain", useOffset=False)
    axins.set_xlabel(r"$x_1$ marginal", fontsize=7, labelpad=1)

    for spine in ["top", "right"]: axins.spines[spine].set_visible(False)

    return fig

def plot_example(output_path="outputs/vi_even_elliptical_example.pdf"):
    left_data = compute_left_panel_data()
    right_data = compute_right_panel_data()
    fig = plot_figure(left_data, right_data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved PDF to {output_path.resolve()}")

    plt.close(fig)

plot_example(output_path="outputs/vi_even_elliptical_example.pdf")