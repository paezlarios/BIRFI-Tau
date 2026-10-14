"""
BIRFI -- Blind Instrument Response Function Identification
=========================================================

Python implementation of

    A. Gomez-Sanchez, O. Devos, R. Vitale, M. Sliwa, D. Sakhapov, J. Enderlein,
    A. de Juan, C. Ruckebusch, "Blind instrument response function
    identification from fluorescence decays", Biophysical Reports 4, 100155
    (2024).  https://doi.org/10.1016/j.bpr.2024.100155

including the supporting information, whose MATLAB listing (`birfi_ls`,
`birfi_dec`) is transcribed here under the same names.

------------------------------------------------------------------------------
The mathematics (SI, Eqs. S1-S6)
------------------------------------------------------------------------------
With x(t) = A exp(-t/tau) for t >= 0 and an IRF of finite support [0, m], the
convolution splits into three domains:

    Domain 1,  t < 0      :  y(t) = 0
    Domain 2,  0 <= t < m :  y(t) = A exp(-t/tau) * int_0^t exp(s/tau) IRF(s) ds
    Domain 3,  t >= m     :  y(t) = A exp(-t/tau) * int_0^m exp(s/tau) IRF(s) ds

In Domain 3 the integral is a *constant* B, so y(t) = B * x(t): past the end of
the IRF support the measured decay is the underlying decay, merely scaled.  Two
consequences drive the algorithm:

  (1) The point where y becomes exponential -- the "cutting point" -- is the end
      of the IRF support, so it hands you the IRF length m for free.
  (2) The tail of y, shifted to the origin, *is* x up to that scalar.  So the
      blind problem "find h and x from y" collapses to the well-posed system
      identification problem "find h from y and x" -- with no need to model x
      at all.  In the published code this is a single call to MATLAB's `deconv`.

Differentiating (Eq. S6) leaves Domain 3 proportional to the original
exponential while distorting Domain 2, which sharpens the frontier; the SI notes
the derivative *minimum* approximates it closely, and is explicit that this is
"a graphical rather than absolute mathematical criterion".

------------------------------------------------------------------------------
Two choices this implementation exposes
------------------------------------------------------------------------------
`x_source`:
    'tail'  -- use the measured tail verbatim (what the published code does).
               Model-free: tau is never estimated.
    'model' -- fit tau to the tail and use a noiseless exp(-t/tau) instead.
               Adds a monoexponential assumption but removes the noise that the
               measured tail otherwise injects into the design matrix.

`cut_method`:
    'derivative_min' -- the SI's graphical criterion, argmin of dy/dt.
    'tail_extrap'    -- fit the clean far tail, walk backwards until the data
                        departs from that extrapolation.
------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.optimize import least_squares, nnls
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

__all__ = [
    "whittaker_smooth",
    "estimate_background",
    "find_cutting_point",
    "fit_tail_monoexp",
    "build_input_signal",
    "deconvolve_irf",
    "birfi_dec",
    "birfi_ls",
    "birfi",
    "reconvolution_fit",
    "select_smoothing",
    "BirfiResult",
    "simulate_decay",
    "gaussian_irf",
    "noise_percent",
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _diff_matrix(n, d):
    """Sparse d-th order difference operator of shape (n-d, n)."""
    D = sparse.eye(n, format="csc")
    for _ in range(d):
        m = D.shape[0]
        Dk = sparse.diags(
            [-np.ones(m - 1), np.ones(m - 1)], [0, 1], shape=(m - 1, m), format="csc"
        )
        D = Dk @ D
    return D.tocsc()


def whittaker_smooth(y, lam=1e2, d=2, w=None):
    """Whittaker smoother (Eilers, Anal. Chem. 75:3631, 2003), ref (20)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    D = _diff_matrix(n, d)
    W = sparse.diags(w, 0, format="csc")
    return spsolve((W + lam * (D.T @ D)).tocsc(), W @ y)


def estimate_background(y, n_pre=None, frac=0.05, use_tail=True):
    """Flat background, from the pre-pulse region and/or the far tail.

    Both regions are only *upper bounds* on the dark level: the pre-pulse bins
    can carry afterpulsing or a reflection, and the far tail still holds signal
    if the decay is incomplete.  Taking the smaller of the two is the safe
    default -- over-subtracting drives the tail negative, which silently wrecks
    any chi-squared computed there.
    """
    y = np.asarray(y, dtype=float)
    cands = []
    if n_pre is None:
        above = np.flatnonzero(y > frac * y.max())
        n_pre = int(above[0]) if above.size and above[0] > 3 else 0
    if n_pre >= 3:
        cands.append(float(np.median(y[:n_pre])))
    if use_tail:
        n_tail = max(5, y.size // 10)
        cands.append(float(np.median(y[-n_tail:])))
    return min(cands) if cands else 0.0


# ---------------------------------------------------------------------------
# Step 1 -- the cutting point (end of the IRF support, "m" in the SI)
# ---------------------------------------------------------------------------
def _cut_derivative_min(t, y, sg_window=None, sg_order=3, margin=0.0):
    """SI criterion: the minimum of dy/dt marks the Domain 2 / Domain 3 frontier.

    Eq. S6 leaves Domain 3 as -(B/tau) A exp(-t/tau), a clean exponential rising
    back towards zero, while Domain 2 is distorted; the frontier therefore shows
    up as the derivative minimum.  Approximate by the SI's own admission, hence
    the `margin` knob.
    """
    dt = float(np.median(np.diff(t)))
    n = y.size
    if sg_window is None:
        sg_window = max(7, int(round(0.015 * n)) | 1)
    sg_window = min(sg_window | 1, (n // 2) * 2 - 1)
    dy = savgol_filter(y, sg_window, sg_order, deriv=1, delta=dt)
    ipk = int(np.argmax(y))
    i_cut = ipk + int(np.argmin(dy[ipk:]))
    return int(round(i_cut * (1.0 + margin))), dy


def _cut_tail_extrapolation(t, y, rel_tol=0.02, frac_min=0.02, persist=3, margin=0.15):
    """Fit the clean far tail, walk backwards to the departure point.

    Equivalent in principle to the derivative criterion (y' is proportional to y
    throughout Domain 3) but tests the deviation directly, which is steadier at
    low photon counts.  Overshooting is cheap -- the extra IRF samples come out
    near zero -- while undershooting truncates the IRF, so the result is padded.
    """
    ipk = int(np.argmax(y))
    thr = frac_min * y.max()
    after = np.flatnonzero(y[ipk:] < thr)
    iend = min(ipk + int(after[0]) if after.size else y.size, y.size)
    if iend - ipk < 20:
        raise ValueError("Tail too short to identify a cutting point.")

    imid = ipk + (iend - ipk) // 2
    seg = slice(imid, iend)
    m = y[seg] > 0
    p = np.polyfit(t[seg][m], np.log(y[seg][m]), 1, w=np.sqrt(y[seg][m]))
    tau_ref = -1.0 / p[0]
    if not np.isfinite(tau_ref) or tau_ref <= 0:
        raise ValueError("Could not estimate a reference lifetime from the tail.")
    ref = np.exp(p[1]) * np.exp(-t / tau_ref)

    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(ref > 0, (y - ref) / ref, np.nan)
    bad = np.abs(dev) > rel_tol
    bad[~np.isfinite(dev)] = True

    i_cut = 0
    for i in range(imid - 1, persist - 1, -1):
        if bad[i - persist + 1 : i + 1].all():
            i_cut = i + 1
            break
    i_cut = int(min(round((i_cut + 1) * (1.0 + margin)), imid) - 1)
    return max(i_cut, ipk + 1), dev


def find_cutting_point(t, y, method="derivative_min", return_profile=False, **kw):
    """Locate the cutting point.  IRF length = i_cut + 1 bins.

    method : {'tail_extrap', 'derivative_min'}
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if method == "derivative_min":
        i_cut, prof = _cut_derivative_min(t, y, **kw)
    elif method == "tail_extrap":
        i_cut, prof = _cut_tail_extrapolation(t, y, **kw)
    else:
        raise ValueError(f"unknown cut method {method!r}")
    i_cut = int(np.clip(i_cut, 2, y.size - 5))
    return (i_cut, prof) if return_profile else i_cut


# ---------------------------------------------------------------------------
# Step 2 -- the input signal x
# ---------------------------------------------------------------------------
def fit_tail_monoexp(t, y, i_cut, i_end=None, refine=True):
    """Fit A*exp(-t/tau) to y[i_cut:i_end].

    Only *used* when x_source='model'; otherwise reported as a diagnostic.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if i_end is None:
        thr = 0.01 * y.max()
        after = np.flatnonzero(y[i_cut:] < thr)
        i_end = i_cut + int(after[0]) if after.size else y.size
    ts, ys = t[i_cut:i_end], y[i_cut:i_end]
    m = ys > 0
    p = np.polyfit(ts[m], np.log(ys[m]), 1, w=np.sqrt(ys[m]))
    tau, A = -1.0 / p[0], np.exp(p[1])
    if refine:

        def resid(par):
            model = np.exp(par[1]) * np.exp(-ts / par[0])
            return (ys - model) / np.sqrt(np.maximum(model, 1.0))

        sol = least_squares(resid, [tau, np.log(A)], method="lm")
        tau, A = sol.x[0], np.exp(sol.x[1])
    return float(tau), float(A)


def build_input_signal(t, y, i_cut, x_source="tail", tau=None):
    """Return the signal x that the decay is deconvolved against.

    'tail'  : y[i_cut:] verbatim -- by Eq. S5 this is x up to a scalar.
              This is what the published MATLAB code uses.
    'model' : a noiseless exp(-t/tau) spanning the whole window.
    """
    if x_source == "tail":
        return np.asarray(y, float)[i_cut:]
    if x_source == "model":
        if tau is None:
            tau, _ = fit_tail_monoexp(t, y, i_cut)
        return np.exp(-(np.asarray(t, float) - t[0]) / tau)
    raise ValueError(f"unknown x_source {x_source!r}")


# ---------------------------------------------------------------------------
# Step 3 -- system identification
# ---------------------------------------------------------------------------
def _long_division(y, x, n):
    """Forward polynomial long division: quotient q with conv(x, q) ~ y.

    Equivalent to MATLAB's `deconv` for these operands.
    """
    q = np.zeros(n)
    r = np.asarray(y, float).copy()
    k = x.size
    for i in range(n):
        q[i] = r[i] / x[0]
        j = min(i + k, r.size)
        r[i:j] -= q[i] * x[: j - i]
    return q


def _conv_design(x, n_rows, n_cols):
    """n_rows x n_cols convolution matrix X with X[n, k] = x[n-k], else 0."""
    col = np.zeros(n_rows)
    m = min(x.size, n_rows)
    col[:m] = x[:m]
    row = np.zeros(n_cols)
    row[0] = col[0]
    return toeplitz(col, row)


def deconvolve_irf(y, x, n_irf, method="lstsq", lam_reg=0.0, weights=None):
    """Recover the IRF from the decay `y` and the input signal `x`.

    method : {'lstsq', 'nnls', 'longdiv'}
        'lstsq'   ordinary least squares on the full convolution system -- the
                  numerically stable form, equivalent to the SI's `birfi_ls`
        'nnls'    same, constrained to a nonnegative IRF
        'longdiv' the SI's `birfi_dec`, i.e. MATLAB `deconv`: exact, but the
                  most noise-sensitive
    lam_reg : second-difference roughness penalty (ignored by 'longdiv')

    Returns (irf normalised to unit area, scale factor).
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    if method == "longdiv":
        h = _long_division(y, x, n_irf)
    else:
        if method not in ("lstsq", "nnls"):
            raise ValueError(f"unknown method {method!r}")
        X = _conv_design(x, y.size, n_irf)
        b = y.copy()
        if weights is not None:
            W = np.asarray(weights, float)
            X, b = X * W[:, None], b * W
        if lam_reg > 0:
            D = np.diff(np.eye(n_irf), 2, axis=0)
            X = np.vstack([X, np.sqrt(lam_reg) * D])
            b = np.concatenate([b, np.zeros(D.shape[0])])
        if method == "nnls":
            h, _ = nnls(X, b)
        else:
            h, *_ = np.linalg.lstsq(X, b, rcond=None)

    area = h.sum()
    if area == 0:
        raise ValueError("Degenerate IRF (zero area).")
    return h / area, float(area)


# ---------------------------------------------------------------------------
# Faithful transcriptions of the SI listing
# ---------------------------------------------------------------------------
def birfi_dec(decay, irf_size):
    """Transcription of the SI's `birfi_dec`.

        decay_cut = decay(1, irf_size:end)
        irf       = deconv(decay, decay_cut)

    `irf_size` is 1-based as in MATLAB: it is the cutting point *and* the
    resulting IRF length.  Returns the unnormalised IRF.
    """
    decay = np.asarray(decay, float)
    decay_cut = decay[irf_size - 1 :]
    return _long_division(decay, decay_cut, decay.size - decay_cut.size + 1)


def birfi_ls(decay, irf_size):
    """Transcription of the SI's `birfi_ls` (Hankel matrix + least squares).

    The MATLAB code pads the cut tail with `irf_size-1` zeros on each side,
    builds a flipped Hankel matrix and solves `irf = decay / HankelM'`.  Working
    the indices through, HankelM(i,j) = decay_cut(i-j+1), so that system is
    exactly conv(decay_cut, irf) = decay; it is assembled here directly as a
    convolution matrix.
    """
    decay = np.asarray(decay, float)
    decay_cut = decay[irf_size - 1 :]
    X = _conv_design(decay_cut, decay.size, irf_size)
    h, *_ = np.linalg.lstsq(X, decay, rcond=None)
    return h


# ---------------------------------------------------------------------------
# Smoothing selection
# ---------------------------------------------------------------------------
def select_smoothing(
    t,
    y,
    i_cut,
    n_irf,
    x_source="tail",
    tau=None,
    grid=None,
    method="nnls",
    lam_reg=0.0,
    factor=0.10,
    floor=1e-3,
):
    """Choose the Whittaker penalty for the deconvolution input automatically.

    BIRFI has a sharp bias/variance trade-off in the pre-smoothing: too little
    and the noise-amplifying deconvolution rings, too much and the IRF is
    artificially broadened.  Each candidate IRF is judged by reconvolving it and
    comparing against the *unsmoothed* data.

    Rule: the largest penalty whose chi-squared is still within `factor` of the
    best achievable -- a relative discrepancy principle sitting on the knee.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    grid = np.geomspace(0.02, 500.0, 18) if grid is None else np.asarray(grid, float)

    chis, lams = [], []
    for lam in grid:
        yd = whittaker_smooth(y, lam=lam) if lam > 0 else y
        try:
            x = build_input_signal(t, yd, i_cut, x_source=x_source, tau=tau)
            irf, amp = deconvolve_irf(yd, x, n_irf, method=method, lam_reg=lam_reg)
            m = amp * np.convolve(irf, x)[: t.size]
        except Exception:
            continue
        sel = m > floor * m.max()
        if sel.sum() < 20:
            continue
        chis.append(np.mean((y[sel] - m[sel]) ** 2 / np.maximum(m[sel], 1.0)))
        lams.append(lam)
    if not chis:
        raise RuntimeError("Smoothing selection failed on every candidate.")
    chis, lams = np.asarray(chis), np.asarray(lams)
    ok = chis <= (1.0 + factor) * chis.min()
    return float(lams[ok][-1]), lams, chis


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
@dataclass
class BirfiResult:
    t: np.ndarray
    t_irf: np.ndarray
    irf: np.ndarray
    tau: float
    amplitude: float
    i_cut: int
    t_cut: float
    background: float
    y_raw: np.ndarray
    y_prep: np.ndarray
    fit: np.ndarray = field(default=None)
    diagnostics: dict = field(default_factory=dict)

    @property
    def fwhm(self):
        """FWHM of the recovered IRF, for comparison with the instrument spec."""
        h = self.irf
        half = h.max() / 2.0
        above = np.flatnonzero(h >= half)
        if above.size < 2:
            return np.nan
        dt = float(np.median(np.diff(self.t)))
        i, j = above[0], above[-1]
        lo = i - (h[i] - half) / (h[i] - h[i - 1]) if i > 0 and h[i] > h[i - 1] else i
        hi = (
            j + (h[j] - half) / (h[j] - h[j + 1])
            if j + 1 < h.size and h[j] > h[j + 1]
            else j
        )
        return float((hi - lo) * dt)

    def __repr__(self):
        return (
            f"BirfiResult(tau={self.tau:.4g}, t_cut={self.t_cut:.4g}, "
            f"n_irf={self.irf.size}, fwhm={self.fwhm:.4g}, "
            f"background={self.background:.4g})"
        )


def birfi(
    t,
    y,
    x_source="tail",
    cut_method="derivative_min",
    lam_detect=1e3,
    lam_deconv="auto",
    subtract_background=True,
    i_cut=None,
    t_cut=None,
    method="lstsq",
    lam_reg=0.0,
    n_irf=None,
    cut_kwargs=None,
):
    """Run the full BIRFI pipeline on one decay.

    Parameters
    ----------
    t, y : (N,) arrays
        Time axis and measured decay.  The window must start *before* the IRF
        rises; the recovered IRF is positioned relative to t[0].
    x_source : {'tail', 'model'}
        'tail' reproduces the published algorithm (no lifetime is estimated).
    cut_method : {'derivative_min', 'tail_extrap'}
        'derivative_min' is the SI's own criterion and is markedly steadier at
        low photon counts.
    lam_detect : Whittaker penalty for cutting-point detection.
    lam_deconv : Whittaker penalty for the deconvolution input, or 'auto'.
    i_cut, t_cut : override the cutting point (the paper fixes it at 2.00 ns for
        the Alexa 647 data rather than detecting it).
    method, lam_reg : passed to `deconvolve_irf`.
    n_irf : override the IRF length; defaults to i_cut + 1.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    bg = estimate_background(y) if subtract_background else 0.0
    y0 = y - bg
    y_det = whittaker_smooth(y0, lam=lam_detect) if lam_detect else y0

    prof = None
    if t_cut is not None:
        i_cut = int(np.searchsorted(t, t_cut))
    elif i_cut is not None:
        i_cut = int(i_cut)
    else:
        i_cut, prof = find_cutting_point(
            t, y_det, method=cut_method, return_profile=True, **(cut_kwargs or {})
        )

    tau, _ = fit_tail_monoexp(t, y0, i_cut)

    L = int(n_irf) if n_irf is not None else i_cut + 1
    L = max(3, min(L, y.size))

    lam_curve = None
    if isinstance(lam_deconv, str):
        if lam_deconv != "auto":
            raise ValueError("lam_deconv must be a number or 'auto'")
        lam_deconv, lg, cg = select_smoothing(
            t, y0, i_cut, L, x_source=x_source, tau=tau, method=method, lam_reg=lam_reg
        )
        lam_curve = (lg, cg)

    y_dec = whittaker_smooth(y0, lam=lam_deconv) if lam_deconv else y0
    x = build_input_signal(t, y_dec, i_cut, x_source=x_source, tau=tau)
    irf, amp = deconvolve_irf(y_dec, x, L, method=method, lam_reg=lam_reg)
    model = amp * np.convolve(irf, x)[: t.size]

    return BirfiResult(
        t=t,
        t_irf=t[:L] - t[0],
        irf=irf,
        tau=tau,
        amplitude=amp,
        i_cut=i_cut,
        t_cut=float(t[i_cut]),
        background=float(bg),
        y_raw=y,
        y_prep=y_dec,
        fit=model,
        diagnostics=dict(
            y_detect=y_det,
            cut_profile=prof,
            x_source=x_source,
            cut_method=cut_method,
            lam_deconv=lam_deconv,
            lam_curve=lam_curve,
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def reconvolution_fit(t, y, irf, n_exp=1, p0=None, fit_background=True):
    """Poisson-weighted reconvolution fit, as used for Fig. 4 of the paper."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    irf = np.asarray(irf, float)
    irf = irf / irf.sum()
    tt = t - t[0]

    if p0 is None:
        span = tt[-1] - tt[0]
        taus0 = np.geomspace(span / 20, span / 4, n_exp)
        amps0 = np.full(n_exp, y.max())
        bg0 = max(np.median(y[:10]), 1e-9)
    else:
        taus0, amps0, bg0 = p0

    def model(par):
        taus = np.abs(par[:n_exp])
        amps = np.abs(par[n_exp : 2 * n_exp])
        bg = np.abs(par[-1]) if fit_background else 0.0
        x = sum(a * np.exp(-tt / tu) for a, tu in zip(amps, taus))
        return np.convolve(irf, x)[: t.size] + bg

    def resid(par):
        m = model(par)
        return (y - m) / np.sqrt(np.maximum(m, 1.0))

    sol = least_squares(
        resid, np.concatenate([taus0, amps0, [bg0]]), method="lm", max_nfev=20000
    )
    m = model(sol.x)
    sel = m > 1e-3 * m.max()
    dof = max(1, int(sel.sum()) - sol.x.size)
    return dict(
        taus=np.abs(sol.x[:n_exp]),
        amps=np.abs(sol.x[n_exp : 2 * n_exp]),
        background=abs(sol.x[-1]) if fit_background else 0.0,
        fit=m,
        weighted_residuals=(y - m) / np.sqrt(np.maximum(m, 1.0)),
        chi2_red=float(np.sum(resid(sol.x)[sel] ** 2) / dof),
        success=bool(sol.success),
    )


# ---------------------------------------------------------------------------
# Simulation (SI: simulated datasets, Eq. S7)
# ---------------------------------------------------------------------------
def gaussian_irf(t, centers, widths, weights=None):
    """Sum of Gaussians, as used for IRF 1 / 2 / 3 (Fig. S3)."""
    centers = np.atleast_1d(np.asarray(centers, float))
    widths = np.atleast_1d(np.asarray(widths, float))
    weights = np.ones_like(centers) if weights is None else np.atleast_1d(weights)
    h = sum(
        w * np.exp(-0.5 * ((t - c) / s) ** 2)
        for c, s, w in zip(centers, widths, weights)
    )
    return h / h.sum()


def noise_percent(y, y_free):
    """Noise metric of Eq. S7: 100 * ||y - y_free|| / ||y_free||."""
    y, y_free = np.asarray(y, float), np.asarray(y_free, float)
    return 100.0 * np.sqrt(np.sum((y - y_free) ** 2) / np.sum(y_free**2))


def simulate_decay(irf, tau, t, noise=None, peak_counts=None, rng=None):
    """Convolve a monoexponential with `irf` and add Poisson noise.

    `noise` is the target percentage in the SI's own metric (Eq. S7).  For
    Poisson counts with mean S*u, E[sum (y-yf)^2] = S*sum(u), so the scale that
    hits a target E is  S = (100/E)^2 * sum(u)/sum(u^2) -- solved exactly here,
    so 0.5 % and 2 % mean what they mean in the paper.

    `peak_counts` is the alternative parameterisation (counts at the maximum).

    Returns (noisy decay, noise-free decay).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.exp(-(t - t[0]) / tau)
    u = np.convolve(irf, x)[: t.size]
    u = u / u.max()
    if noise is not None:
        S = (100.0 / noise) ** 2 * u.sum() / np.sum(u**2)
        return rng.poisson(u * S).astype(float), u * S
    if peak_counts is not None:
        return rng.poisson(u * peak_counts).astype(float), u * peak_counts
    return u.copy(), u.copy()
