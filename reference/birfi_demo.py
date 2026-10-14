"""Reproduce the key figures of Gomez-Sanchez et al., Biophys. Rep. 4, 100155 (2024)
and its supporting information.

Synthetic setup follows the SI: monoexponential x with tau = 1 ns, 1498 sampling
points at 25 ps, convolved with three IRF shapes (Fig. S3: one, two and three
Gaussians), with Poisson noise calibrated to the SI's own metric (Eq. S7).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from birfi import birfi, gaussian_irf, reconvolution_fit, simulate_decay

plt.rcParams.update(
    {"figure.dpi": 130, "font.size": 8, "axes.grid": True, "grid.alpha": 0.25}
)

DT, NBIN, TAU = 0.025, 1498, 1.0  # ns
T = np.arange(NBIN) * DT

SHAPES = {
    "IRF 1": ([0.60], [0.09], [1.0]),
    "IRF 2": ([0.55, 0.80], [0.09, 0.09], [1.0, 0.6]),
    "IRF 3": ([0.50, 0.72, 0.95], [0.08, 0.08, 0.08], [1.0, 0.8, 0.45]),
}
NOISE = [("Noise 0%", None), ("Noise 0.5%", 0.5), ("Noise 2%", 2.0)]


def make(key, noise, seed=0):
    c, s, w = SHAPES[key]
    irf_true = gaussian_irf(T, c, s, w)
    rng = np.random.default_rng(seed)
    if noise is None:
        y, _ = simulate_decay(irf_true, TAU, T, peak_counts=None, rng=rng)
        y = y * 1e4
    else:
        y, _ = simulate_decay(irf_true, TAU, T, noise=noise, rng=rng)
    return irf_true, y


# ------------------------------------------------- Figure 1 (+ Fig. S2) ------
def figure1(path="fig1_principle.png"):
    irf_true, y = make("IRF 3", 0.5, seed=1)
    res = birfi(T, y, method="nnls")
    dy = savgol_filter(res.diagnostics["y_detect"], 23, 3, deriv=1, delta=DT)
    ic = res.i_cut

    fig, ax = plt.subplots(2, 2, figsize=(7.6, 5.0))
    a = ax[0, 0]
    a.plot(T, y, "k", lw=0.8)
    a.set(xlim=(0, 10), xlabel="Time (ns)", ylabel="Photon counts",
          title="A  measured decay $y$")

    a = ax[0, 1]
    a.plot(T[:ic], dy[:ic], "k", lw=1.0, label="Domain 2 (IRF-distorted)")
    a.plot(T[ic:], dy[ic:], "r", lw=1.0, label="Domain 3 (exponential)")
    a.plot(T[ic], dy[ic], "o", ms=4, mfc="none", mec="0.2")
    a.annotate("optimal point\n(derivative minimum)", (T[ic], dy[ic]),
               (T[ic] + 1.4, dy.min() * 0.55),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=6.5)
    a.set(xlim=(0, 10), xlabel="Time (ns)", ylabel=r"$\partial y/\partial t$",
          title="B  first derivative (Eq. S6)")
    a.legend(fontsize=6.5, loc="upper right")

    a = ax[1, 0]
    a.plot(T, y, "k", lw=0.8)
    a.plot(T[ic:], y[ic:], "r", lw=1.2, label=r"tail $=x$ up to a scalar (Eq. S5)")
    a.axvline(res.t_cut, color="0.4", ls="--", lw=0.8)
    a.text(res.t_cut + 0.15, y.max() * 0.75, "$m$", fontsize=9)
    a.set(xlim=(0, 10), xlabel="Time (ns)", ylabel="Photon counts",
          title="C  the tail is the input signal")
    a.legend(fontsize=6.5)

    a = ax[1, 1]
    n = res.irf.size
    a.plot(T[:n], irf_true[:n] / irf_true[:n].max(), "k", lw=1.4, label="true IRF")
    a.plot(T[:n], res.irf / res.irf.max(), "r--", lw=1.2, label="BIRFI estimate")
    a.set(xlim=(0, 2), xlabel="Time (ns)", ylabel="Normalised intensity",
          title="D  system identification")
    a.legend(fontsize=6.5)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, res


# ------------------------------------------------------------- Figure 2 -----
def figure2(n_rep=40, path="fig2_montecarlo.png"):
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 6.4), sharex=True)
    summary = []
    for i, key in enumerate(SHAPES):
        for j, (label, E) in enumerate(NOISE):
            irf_true, _ = make(key, E, seed=0)
            reps, errs = [], []
            for s in range(n_rep):
                _, y = make(key, E, seed=1000 * j + s)
                try:
                    r = birfi(T, y, method="nnls")
                except Exception:
                    continue
                h = np.zeros(120)
                m = min(120, r.irf.size)
                h[:m] = r.irf[:m]
                reps.append(h / h.max())
                errs.append(np.linalg.norm(r.irf[:m] - irf_true[:m])
                            / np.linalg.norm(irf_true[:m]))
                if E is None:
                    break
            R = np.array(reps)
            mu, sd = R.mean(0), R.std(0)
            tt = T[:120]
            a = axes[i, j]
            a.fill_between(tt, mu - 1.96 * sd, mu + 1.96 * sd, color="r", alpha=0.3, lw=0)
            a.plot(tt, mu, "r", lw=1.1)
            a.plot(tt, irf_true[:120] / irf_true[:120].max(), "k--", lw=1.0)
            a.set_xlim(0, 1.8)
            if i == 0:
                a.set_title(label, fontsize=8)
            if j == 0:
                a.set_ylabel(f"{key}\nNormalised intensity", fontsize=7.5)
            if i == 2:
                a.set_xlabel("Time (ns)")
            a.text(0.03, 0.93, f"rel. err {np.mean(errs):.3f}", transform=a.transAxes,
                   fontsize=6.5, va="top")
            summary.append((key, label, float(np.mean(errs))))
    fig.suptitle("BIRFI recovery: mean (red) with 95% interval vs ground truth (black)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, summary


# ------------------------------------------------------------- Figure 4 -----
def figure4(path="fig4_residuals.png"):
    """Reconvolution residuals: BIRFI IRF vs a mismatched 'measured' IRF.

    The mismatch emulates the colour effect the paper highlights: the IRF is
    recorded by scattering at the laser wavelength while the decay is detected
    in the red, so the true IRF is shifted and broadened relative to it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharex=True)
    rows = []
    for k, key in enumerate(SHAPES):
        c, s, w = SHAPES[key]
        irf_true = gaussian_irf(T, c, s, w)
        irf_meas = gaussian_irf(T, [v - 0.04 for v in c], [v * 0.85 for v in s], w)
        y, _ = simulate_decay(irf_true, TAU, T, noise=0.5,
                              rng=np.random.default_rng(7 + k))
        res = birfi(T, y, method="nnls")
        n = res.irf.size
        fb = reconvolution_fit(T, y, res.irf, n_exp=1)
        fm = reconvolution_fit(T, y, irf_meas[:n], n_exp=1)
        a = axes[k]
        a.plot(T, fm["weighted_residuals"], color="0.25", lw=0.5, label="measured IRF")
        a.plot(T, fb["weighted_residuals"], color="crimson", lw=0.5, alpha=0.8,
               label="BIRFI IRF")
        a.axhline(0, color="k", lw=0.6)
        a.set(xlim=(0, 12), xlabel="Time (ns)", title=key)
        if k == 0:
            a.set_ylabel("Weighted residuals")
            a.legend(fontsize=6.5, loc="upper right")
        rows.append((key, fb["taus"][0], fb["chi2_red"], fm["taus"][0], fm["chi2_red"]))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, rows


# ------------------------------------------------ Figure 5 (this work) ------
def figure5(n_rep=15, path="fig5_tail_vs_model.png"):
    """Measured tail (published) vs fitted monoexponential as the input signal."""
    levels = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), sharey=True)
    rows = []
    for k, key in enumerate(SHAPES):
        c, s, w = SHAPES[key]
        irf_true = gaussian_irf(T, c, s, w)
        curves = {}
        for xs in ("tail", "model"):
            means = []
            for E in levels:
                e = []
                for seed in range(n_rep):
                    y, _ = simulate_decay(irf_true, TAU, T, noise=E,
                                          rng=np.random.default_rng(seed))
                    try:
                        r = birfi(T, y, x_source=xs, method="nnls")
                        n = r.irf.size
                        e.append(np.linalg.norm(r.irf - irf_true[:n])
                                 / np.linalg.norm(irf_true[:n]))
                    except Exception:
                        e.append(np.nan)
                means.append(np.nanmean(e))
            curves[xs] = means
        a = axes[k]
        a.plot(levels, curves["tail"], "o-", ms=3.5, color="crimson",
               label="x = measured tail (published)")
        a.plot(levels, curves["model"], "s-", ms=3.5, color="steelblue",
               label=r"x = fitted $e^{-t/\tau}$")
        a.set(xscale="log", yscale="log", xlabel="Noise, Eq. S7 (%)", title=key)
        if k == 0:
            a.set_ylabel("Relative IRF error")
            a.legend(fontsize=6.5)
        rows.append((key, levels, curves))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, rows


if __name__ == "__main__":
    p1, res = figure1()
    print(f"[Fig 1] {p1}  t_cut={res.t_cut:.3f} ns  L={res.irf.size} bins  "
          f"fwhm={res.fwhm*1000:.0f} ps  lambda={res.diagnostics['lam_deconv']:.3g}")

    p4, rows = figure4()
    print(f"\n[Fig 4] {p4}")
    print(f"{'':7s} {'tau BIRFI':>10s} {'chi2red':>9s} {'tau meas':>10s} {'chi2red':>9s}")
    for r in rows:
        print(f"{r[0]:7s} {r[1]:10.4f} {r[2]:9.3f} {r[3]:10.4f} {r[4]:9.3f}")

    p2, summary = figure2()
    print(f"\n[Fig 2] {p2}")
    for key, label, err in summary:
        print(f"  {key:6s} {label:11s} mean relative IRF error = {err:.4f}")

    p5, rows5 = figure5()
    print(f"\n[Fig 5] {p5}")
    for key, lv, cv in rows5:
        print(f"  {key}: " + "  ".join(
            f"{E:g}%={cv['tail'][i]:.3f}/{cv['model'][i]:.3f}" for i, E in enumerate(lv)))
