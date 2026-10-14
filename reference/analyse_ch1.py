"""BIRFI analysis of LifeTimeHistogramCh1.csv: extract the IRF, then fit tau."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from birfi import birfi, estimate_background, reconvolution_fit

plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25})

CSV = "/mnt/user-data/uploads/LifeTimeHistogramCh1.csv"
T_START = 1.0          # crop: start just before the rise, keeps the IRF compact


def load():
    d = np.loadtxt(CSV, delimiter=",", skiprows=1)
    t, y = d[:, 0], d[:, 1]
    k = t >= T_START
    return t[k], y[k]


def analyse(t, y, **kw):
    r = birfi(t, y, method="nnls", **kw)
    f = reconvolution_fit(t, y, r.irf, n_exp=1)
    return r, f


if __name__ == "__main__":
    t, y = load()
    bg = estimate_background(y)
    r, f = analyse(t, y)
    tau = f["taus"][0]

    print(f"bins {t.size}, dt {np.median(np.diff(t))*1000:.0f} ps, "
          f"peak {y.max():.0f} counts, background {bg:.1f}")
    print(f"\nIRF   : support {r.irf.size} bins = {r.irf.size*0.1:.1f} ns, "
          f"FWHM {r.fwhm*1000:.0f} ps, peak at {t[np.argmax(r.irf)]:.2f} ns")
    print(f"        cutting point {r.t_cut:.2f} ns, lambda {r.diagnostics['lam_deconv']:.4g}")
    print(f"Fit   : tau = {tau:.4f} ns, chi2_red = {f['chi2_red']:.3f}")

    # ---- robustness: does the answer depend on my choices?
    print("\nrobustness")
    for lbl, kw in [("default", {}),
                    ("x = measured tail", dict(x_source="tail")),
                    ("x = fitted exp", dict(x_source="model")),
                    ("cut: tail_extrap", dict(cut_method="tail_extrap")),
                    ("cut forced 3.5 ns", dict(t_cut=3.5)),
                    ("cut forced 5.0 ns", dict(t_cut=5.0)),
                    ("solver: lstsq", dict(method_="lstsq")),
                    ("no smoothing", dict(lam_deconv=0.0))]:
        kk = dict(kw)
        meth = kk.pop("method_", "nnls")
        rr = birfi(t, y, method=meth, **kk)
        ff = reconvolution_fit(t, y, rr.irf, n_exp=1)
        print(f"  {lbl:20s} fwhm {rr.fwhm*1000:5.0f} ps   tau {ff['taus'][0]:.4f} ns   "
              f"chi2 {ff['chi2_red']:.3f}")

    for ts in (0.0, 0.5, 1.5, 2.0):
        d = np.loadtxt(CSV, delimiter=",", skiprows=1)
        k = d[:, 0] >= ts
        rr, ff = analyse(d[k, 0], d[k, 1])
        print(f"  {'crop t>=%.1f ns' % ts:20s} fwhm {rr.fwhm*1000:5.0f} ps   "
              f"tau {ff['taus'][0]:.4f} ns   chi2 {ff['chi2_red']:.3f}")

    # ---- parametric bootstrap for the uncertainty on tau
    rng = np.random.default_rng(0)
    model = np.maximum(f["fit"], 0)
    taus, fwhms = [], []
    for _ in range(120):
        yb = rng.poisson(model).astype(float)
        try:
            rb, fb = analyse(t, yb)
            taus.append(fb["taus"][0])
            fwhms.append(rb.fwhm * 1000)
        except Exception:
            pass
    taus, fwhms = np.array(taus), np.array(fwhms)
    print(f"\nbootstrap ({taus.size} resamples)")
    print(f"  tau  = {tau:.3f} +- {taus.std():.3f} ns   "
          f"(95% CI {np.percentile(taus,2.5):.3f} - {np.percentile(taus,97.5):.3f})")
    print(f"  FWHM = {r.fwhm*1000:.0f} +- {fwhms.std():.0f} ps")

    # tail-only fit, which needs no IRF at all, as a cross-check
    from birfi import fit_tail_monoexp
    tt, _ = fit_tail_monoexp(t, y - bg, int(np.searchsorted(t, 5.0)),
                             int(np.searchsorted(t, 13.0)))
    print(f"\ncross-check, tail-only fit (5-13 ns, no IRF): tau = {tt:.4f} ns")

    # ---- figure
    fig = plt.figure(figsize=(8.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.25)

    a = fig.add_subplot(gs[0, 0])
    a.semilogy(t, y, ".", ms=2, color="0.45", label="data")
    a.semilogy(t, f["fit"], "crimson", lw=1.2, label=f"fit, $\\tau$ = {tau:.3f} ns")
    a.set(ylabel="Photon counts", ylim=(10, 1e5), title="A  reconvolution fit")
    a.set_xticklabels([])
    a.legend(fontsize=7)

    b = fig.add_subplot(gs[1, 0], sharex=a)
    b.plot(t, f["weighted_residuals"], lw=0.6, color="0.25")
    b.axhline(0, color="k", lw=0.6)
    b.set(xlabel="Time (ns)", ylabel="Weighted\nresiduals", ylim=(-5, 5))

    c = fig.add_subplot(gs[:, 1])
    n = r.irf.size
    c.plot(t[:n], r.irf / r.irf.max(), "crimson", lw=1.4)
    c.fill_between(t[:n], 0, r.irf / r.irf.max(), color="crimson", alpha=0.15)
    c.plot(t, y / y.max(), color="0.6", lw=0.8, label="decay (normalised)")
    c.axhline(0.5, color="0.7", ls=":", lw=0.8)
    c.set(xlim=(1.0, 5.0), xlabel="Time (ns)", ylabel="Normalised intensity",
          title=f"B  BIRFI IRF, FWHM {r.fwhm*1000:.0f} ps")
    c.legend(fontsize=7)

    fig.savefig("ch1_analysis.png", bbox_inches="tight")
    print("\nwrote ch1_analysis.png")
