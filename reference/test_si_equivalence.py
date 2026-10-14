"""Regression tests against the SI's published MATLAB listing and equations."""

import numpy as np

from birfi import (birfi_dec, birfi_ls, deconvolve_irf, gaussian_irf,
                   noise_percent, simulate_decay)

DT, N, TAU = 0.025, 1498, 1.0
T = np.arange(N) * DT
IRF = gaussian_irf(T, [0.50, 0.72, 0.95], [0.08] * 3, [1.0, 0.8, 0.45])
M = int(np.flatnonzero(IRF > 1e-9 * IRF.max())[-1]) + 1


def test_si_functions_agree():
    """The SI states both functions yield equivalent outputs."""
    yf, _ = simulate_decay(IRF, TAU, T)
    for size in (M, M + 20, 60):
        assert np.allclose(birfi_dec(yf, size), birfi_ls(yf, size), atol=1e-8)


def test_noise_free_recovery_is_exact():
    """Eq. S5: with the true support, recovery is a well-posed identification."""
    yf, _ = simulate_decay(IRF, TAU, T)
    h = birfi_dec(yf, M)
    h /= h.sum()
    assert np.linalg.norm(h - IRF[: h.size]) / np.linalg.norm(IRF[: h.size]) < 1e-6


def test_module_solvers_match_si():
    """'longdiv' and 'lstsq' reproduce birfi_dec / birfi_ls."""
    yf, _ = simulate_decay(IRF, TAU, T)
    x = yf[M - 1 :]
    for meth, ref in (("longdiv", birfi_dec), ("lstsq", birfi_ls)):
        h, _ = deconvolve_irf(yf, x, M, method=meth)
        r = ref(yf, M)
        assert np.allclose(h, r / r.sum(), atol=1e-8)


def test_eq_s7_noise_calibration():
    """simulate_decay(noise=E) realises E in the SI's own metric."""
    for target in (0.5, 2.0):
        got = [
            noise_percent(*simulate_decay(IRF, TAU, T, noise=target,
                                          rng=np.random.default_rng(s)))
            for s in range(100)
        ]
        assert abs(np.mean(got) - target) < 0.05 * target


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS  {fn.__name__}")
