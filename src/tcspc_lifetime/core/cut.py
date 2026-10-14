"""
The cut m: where the IRF has finished acting and the tail is the pure decay.

Role in the pipeline: gives the starting cut m0 and the grid of cuts the λ-m scan tries.
Depends on: NumPy and core.smoothing.
"""
import numpy as np

from .smoothing import lambda_for_cutoff, whittaker_smooth


def find_initial_cut(y, rise_bins, smoothing_cutoff_factor=3.0):
    """
    Starting cut m0: the minimum of the derivative after the peak.

    Why:    from the cut onwards the decay equals the scaled underlying exponential (SI Eq.
            S5); the derivative minimum approximates that frontier (SI Eq. S6). It is only a
            starting point: the λ-m scan tries cuts around it. The derivative is taken on a
            trace smoothed with a cutoff of smoothing_cutoff_factor x the rise time - heavy
            enough to locate the minimum reliably, which is all that is needed here.
    Input:  y         - counts above background (analysis window)
            rise_bins - rise time (bins)
    Output: m0 - bin index of the cut within the window
    Used by: make_m_grid.
    """
    smoothed = whittaker_smooth(y, lambda_for_cutoff(smoothing_cutoff_factor * rise_bins))
    derivative = np.gradient(smoothed)                   # counts per bin, per bin
    peak_bin = int(np.argmax(smoothed))
    return peak_bin + int(np.argmin(derivative[peak_bin:]))


def make_m_grid(m0, n_bins, n_before=6, n_after=6, min_tail_bins=10):
    """
    The cuts tried by the λ-m scan: from m0 - n_before to m0 + n_after.

    Why:    the derivative minimum is only approximate. Too early a cut truncates the IRF;
            too late a cut lets the surplus IRF samples absorb noise.
    Input:  m0      - starting cut (bin index)
            n_bins  - number of bins in the window
            n_before, n_after, min_tail_bins - from the configuration
    Output: array of cut indices (never below 1, always leaving min_tail_bins of tail)
    Used by: per_decay.analyse.
    """
    first = max(m0 - n_before, 1)
    last = min(m0 + n_after, n_bins - min_tail_bins)
    return np.arange(first, last + 1)
