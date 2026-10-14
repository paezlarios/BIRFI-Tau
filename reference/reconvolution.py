"""
Reconvolution fit of a TCSPC decay with a known (e.g. BIRFI) IRF.  NumPy only.

    model[n] = A * (IRF (*) exp(-t/tau))[n] + b

* Same discrete convolution and time origin as the IRF extraction: no resampling, no shift.
* Variable projection: A, b are linear and solved exactly (A, b >= 0); only tau is searched.
* Poisson maximum likelihood (stat="poisson", default) via IRLS: for an identity link,
  ML = weighted least squares with weights 1/mu, iterated to self-consistency.
  stat="neyman" (weights 1/counts) is kept for comparison: biased low when bins have few counts.
* Error: where the profile deviance rises by 1 (A and b re-solved at every tau).

    fit = fit_tau(t, counts, irf)      ->  dict(tau, err, stat_per_dof, A, bg)
"""
import numpy as np

def golden(fun, lo, hi, tol=1e-6):
    """Minimise a 1-D unimodal function on [lo, hi] (golden-section search, numpy only)."""
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g*(b - a), a + g*(b - a)
    fc, fd = fun(c), fun(d)
    while b - a > tol:
        if fc < fd: b, d, fd = d, c, fc; c = b - g*(b - a); fc = fun(c)
        else:       a, c, fc = c, d, fd; d = a + g*(b - a); fd = fun(d)
    return (a + b) / 2

def minimise_1d(fun, lo, hi, n_coarse=40, tol=1e-6):
    """Coarse log grid to find the right valley, then golden-section inside it."""
    grid = np.geomspace(lo, hi, n_coarse)
    k = int(np.argmin([fun(s) for s in grid]))
    return golden(fun, grid[max(k - 1, 0)], grid[min(k + 1, n_coarse - 1)], tol)

def basis(t, irf, tau):
    """Columns of the linear part: IRF (*) exp(-t/tau), and a flat background."""
    g = np.convolve(irf, np.exp(-(t - t[0]) / tau))[:t.size]
    return np.column_stack([g, np.ones(t.size)])

def wls_nonneg(M, y, w):
    """Weighted least squares for (A, b) with A, b >= 0 (exact 2-variable active set)."""
    Mw, yw = M * w[:, None], y * w
    c = np.linalg.lstsq(Mw, yw, rcond=None)[0]
    if c[1] < 0:                                   # background pinned at zero
        c = np.array([max(Mw[:, 0] @ yw / (Mw[:, 0] @ Mw[:, 0]), 0.0), 0.0])
    if c[0] < 0:                                   # amplitude pinned at zero
        c = np.array([0.0, max(Mw[:, 1] @ yw / (Mw[:, 1] @ Mw[:, 1]), 0.0)])
    return c

def linear_neyman(M, y):
    w = 1 / np.sqrt(np.maximum(y, 1))
    c = wls_nonneg(M, y, w)
    return c, np.sum(((y - M @ c) * w)**2)

def linear_poisson(M, y, iters=8):
    """Poisson ML for (A, b) at fixed tau. For an identity link, ML is weighted least squares
    with weights 1/mu, iterated until the weights match the model they produce (IRLS)."""
    c, _ = linear_neyman(M, y)
    for _ in range(iters):
        mu = np.maximum(M @ c, 1e-12)
        c = wls_nonneg(M, y, 1 / np.sqrt(mu))
    mu = np.maximum(M @ c, 1e-12)
    pos = y > 0
    dev = 2 * (np.sum(mu - y) + np.sum(y[pos] * np.log(y[pos] / mu[pos])))
    return c, dev

def fit_tau(t, y, irf, stat="poisson", lo=0.1, hi=20.0, error=True):
    lin = linear_poisson if stat == "poisson" else linear_neyman
    prof = lambda tau: lin(basis(t, irf, tau), y)[1]          # profile: A, b solved exactly
    tau = minimise_1d(prof, lo, hi)
    s0 = prof(tau)
    err = np.nan
    if error:                                                 # 1-sigma where the profile rises by 1
        up = golden(lambda s: abs(prof(s) - s0 - 1), tau, tau * 2.0, 1e-7)
        dn = golden(lambda s: abs(prof(s) - s0 - 1), tau / 2.0, tau, 1e-7)
        err = (up - dn) / 2
    c, _ = lin(basis(t, irf, tau), y)
    return dict(tau=tau, err=err, stat_per_dof=s0 / (t.size - 3), A=c[0], bg=c[1])
