"""
BIRFI: the IRF from a monoexponential decay and its own tail.

Role in the pipeline: turns a smoothed decay and a cut m into an IRF.
Depends on: NumPy.
"""
import numpy as np


def deconvolve_irf(y_smoothed, m):
    """
    The BIRFI deconvolution, by polynomial long division (identical to MATLAB's deconv).

    Why:    for t >= t[m] the IRF has finished acting, so the tail y[m:] is the underlying
            decay x up to a scale (SI Eq. S5). Then y = IRF ⊛ x, and knowing y and x the IRF
            follows by long division. Each division step: what is left of y at bin n can
            only come from IRF sample n, so h[n] = remainder[n] / x[0]; that sample's copy of
            the tail is then subtracted. Bin m is shared: last IRF bin and first tail bin.
    Input:  y_smoothed - counts above background, smoothed
            m          - cut (bin index within the window)
    Output: raw IRF, m + 1 values (bins 0..m of the window), not normalised
    Used by: normalise_irf.
    """
    tail = y_smoothed[m:]                         # the underlying decay x, up to a scale
    irf_raw, _remainder = np.polydiv(y_smoothed, tail)
    return irf_raw


def normalise_irf(irf_raw, clip_negative=True):
    """
    Unit-area IRF.

    Why:    BIRFI leaves the IRF's overall scale undetermined (the fit's amplitude absorbs
            it), so it is normalised to unit area. Negative samples are clipped in both
            modes: in calibration mode, choosing λ and m without clipping gave a τ₂ bias of
            -7.0 ± 1.9 ps (prompt, sections 3 and 4). Clipping itself biases τ by about
            -4 ps only when the IRF is back to zero by the cut (prompt, section 9).
    Input:  irf_raw       - output of deconvolve_irf
            clip_negative - set negative samples to zero first
    Output: IRF with unit area, or None if its area is not positive (the pair is skipped)
    Used by: the λ-m scan and the final result.
    """
    irf = np.clip(irf_raw, 0.0, None) if clip_negative else np.array(irf_raw, dtype=float)
    total = irf.sum()
    return irf / total if total > 0 else None
