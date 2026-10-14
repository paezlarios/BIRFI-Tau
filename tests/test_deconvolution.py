"""BIRFI deconvolution: equivalence with the SI's MATLAB code, and exact recovery."""
import numpy as np

from tcspc_lifetime.core.deconvolution import deconvolve_irf
from tcspc_lifetime.simulate import expected_decay, poisson_decay, two_peak_irf


# --- transcriptions of the SI's MATLAB functions (1-based irf_size, as in MATLAB) ---
def birfi_dec(decay, irf_size):
    """decay_cut = decay(1, irf_size:end); irf = deconv(decay, decay_cut)"""
    decay_cut = decay[irf_size - 1:]
    n = decay.size - decay_cut.size + 1
    irf, remainder = np.zeros(n), decay.astype(float).copy()
    for k in range(n):                                   # MATLAB deconv: long division
        irf[k] = remainder[k] / decay_cut[0]
        remainder[k:k + decay_cut.size] -= irf[k] * decay_cut
    return irf


def birfi_ls(decay, irf_size):
    """The Hankel least-squares system of the SI, i.e. conv(decay_cut, irf) = decay."""
    decay_cut = decay[irf_size - 1:]
    design = np.zeros((decay.size, irf_size))
    for j in range(irf_size):
        design[j:j + decay_cut.size, j] = decay_cut[: decay.size - j]
    return np.linalg.lstsq(design, decay, rcond=None)[0]


TIME_NS = np.arange(250) * 0.1


def compact_irf():
    """Two-peak IRF set exactly to zero after bin 35, so the identity holds exactly."""
    irf = two_peak_irf(TIME_NS)
    irf[36:] = 0.0
    return irf / irf.sum()


def test_si_equivalence_noise_free():
    """Acceptance: np.polydiv = birfi_dec = birfi_ls to within 1e-8 of the IRF peak."""
    y = expected_decay(compact_irf(), TIME_NS, 1.2, 5e4)
    for m in (35, 40, 50):
        ours = deconvolve_irf(y, m)
        peak = np.abs(ours).max()
        assert np.abs(ours - birfi_dec(y, m + 1)).max() < 1e-8 * peak
        assert np.abs(ours - birfi_ls(y, m + 1)).max() < 1e-8 * peak


def test_si_equivalence_noisy():
    """On noisy data, long division (ours and MATLAB's deconv) stays identical."""
    y = poisson_decay(expected_decay(compact_irf(), TIME_NS, 1.2, 5e4), seed=1)
    for m in (25, 35):
        ours = deconvolve_irf(y, m)
        assert np.abs(ours - birfi_dec(y, m + 1)).max() < 1e-8 * np.abs(ours).max()


def test_noise_free_recovery_at_true_cut():
    """Acceptance: relative IRF error below 1e-6 when the cut is where the IRF ends."""
    irf_true = compact_irf()
    y = expected_decay(irf_true, TIME_NS, 1.2, 5e4)
    recovered = deconvolve_irf(y, 35)
    recovered /= recovered.sum()
    error = np.linalg.norm(recovered - irf_true[:36]) / np.linalg.norm(irf_true[:36])
    assert error < 1e-6
