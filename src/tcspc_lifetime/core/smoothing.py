"""
Whittaker smoothing, the relation between λ and a cutoff period, and the λ grid.

Role in the pipeline: smooths the decay before the derivative (starting cut) and before
the deconvolution (one smoothing per λ of the λ-m scan).
Depends on: NumPy, whittaker-eilers.
"""
import numpy as np
from whittaker_eilers import WhittakerSmoother


def lambda_for_cutoff(cutoff_bins):
    """
    Whittaker penalty λ whose filter passes half the amplitude at the given period.

    Why:    the order-2 smoother has gain H(ω) = 1/(1 + 16 λ sin^4(ω/2)). Setting H = 1/2 at
            ω = 2π/T_c gives the exact relation λ = (2 sin(π/T_c))^-4, valid for T_c >= 2
            bins (Nyquist). A cutoff period is easy to reason about; λ (bins^4) is not.
    Input:  cutoff_bins - period at which the gain is 1/2 (bins, >= 2)
    Output: λ (bins^4)
    Used by: cut.find_initial_cut and make_lambda_grid.
    """
    cutoff_bins = max(cutoff_bins, 2.0)
    return (2.0 * np.sin(np.pi / cutoff_bins)) ** -4


def cutoff_for_lambda(lam):
    """
    Inverse of lambda_for_cutoff: the half-gain period (bins) of a given λ.

    Input:  lam - λ (bins^4)
    Output: cutoff period in bins, or None when λ is too small to halve any period
            (including λ = 0, which does not smooth)
    Used by: results and reports, to state λ in an understandable unit.
    """
    if lam <= 0:
        return None
    half_gain_sine = 0.5 * lam ** -0.25
    return float(np.pi / np.arcsin(half_gain_sine)) if half_gain_sine < 1 else None


def whittaker_smooth(y, lam):
    """
    Whittaker smoother: the curve z that minimises ||y - z||^2 + λ ||D2 z||^2.

    Why:    the deconvolution and the derivative both amplify high-frequency noise; this
            removes it first. Setting the gradient to zero gives (I + λ D2ᵀD2) z = y, which
            the whittaker-eilers package solves exactly (sparse LDLᵀ). λ is always chosen
            here; the package's smooth_optimal is never used (prompt, section 9).
    Input:  y   - data to smooth (counts above background)
            lam - λ (bins^4); λ = 0 returns y unchanged (the package needs λ > 0)
    Output: smoothed copy of y
    Used by: cut.find_initial_cut and the λ-m scan.
    """
    y = np.asarray(y, dtype=float)
    if lam == 0:
        return y.copy()
    smoother = WhittakerSmoother(lmbda=float(lam), order=2, data_length=y.size)
    return np.array(smoother.smooth(y.tolist()))


def make_lambda_grid(rise_bins, top_cutoff_factor=1.5, n_values=13, n_decades=3.0):
    """
    The λ values tried by the λ-m scan.

    Why:    the best λ depends on the noise and the IRF shape, so it is found by trying.
            λ = 0 (no smoothing) is always included; the others are log-spaced over
            n_decades below the λ whose cutoff is top_cutoff_factor x the rise time, so the
            grid stays tied to the IRF width in bins whatever the binning.
    Input:  rise_bins - rise time (bins); the other inputs come from the configuration
    Output: array of λ values (bins^4), starting with 0
    Used by: per_decay.analyse.
    """
    lam_top = lambda_for_cutoff(top_cutoff_factor * rise_bins)
    return np.concatenate([[0.0], np.geomspace(lam_top / 10 ** n_decades, lam_top, n_values)])
