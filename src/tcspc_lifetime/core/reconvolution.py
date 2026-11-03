"""
The reconvolution model and its Poisson maximum-likelihood fit.

Role in the pipeline: scores every (λ, m) pair and gives the final lifetime. The model is
A·(IRF ⊛ exp(-t/τ)) + b; its linear parameters are solved exactly for each trial lifetime
(variable projection) and the lifetime is found by minimising the resulting deviance.
Depends on: NumPy, SciPy.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls


def decay_shape(time_ns, irf, tau_ns):
    """
    Model decay shape for a trial lifetime: IRF ⊛ exp(-t/τ), unit amplitude.

    Why:    what the decay looks like if τ is the lifetime. It uses the same discrete
            convolution and time origin as the deconvolution, so the IRF needs no shift when
            it came from this very decay.
    Input:  time_ns - window time axis (ns)
            irf     - unit-area IRF (bins 0..m of the window)
            tau_ns  - trial lifetime (ns)
    Output: shape, one value per bin of the window
    Used by: profile_deviance, the final model and the Fisher error.
    """
    exponential = np.exp(-(time_ns - time_ns[0]) / tau_ns)
    return np.convolve(irf, exponential)[: time_ns.size]


def poisson_deviance(counts, model):
    """
    Poisson deviance  D = 2 Σ [model - counts + counts·ln(counts/model)].

    Why:    -2 x the log-likelihood of the counts, relative to a model matching every bin
            exactly; minimising D is maximum likelihood. For a fit at the noise level
            D/dof ≈ 1. Bins with zero counts contribute only their model value.
    Input:  counts - raw photons per bin; model - expected photons per bin
    Output: D (dimensionless)
    Used by: every fit and score.
    """
    model = np.maximum(model, 1e-12)
    log_term = np.zeros_like(model)
    has_counts = counts > 0
    log_term[has_counts] = counts[has_counts] * np.log(counts[has_counts] / model[has_counts])
    return 2.0 * np.sum(model - counts + log_term)


def fit_linear_parameters(columns, counts, max_iterations=30, fixed_part=None):
    """
    Poisson maximum-likelihood values of the parameters that enter the model linearly.

    Why:    with model = Σ_k c_k·column_k, the likelihood equations are
            Σ_i column_k[i]·(counts_i - model_i)/model_i = 0. Weighted least squares with
            weights 1/model solves the same equations, but the weights depend on the answer,
            so the solve is repeated until nothing changes (iteratively reweighted least
            squares); the fixed point is the maximum-likelihood solution. The likelihood is
            concave in these parameters, so there is exactly one maximum. scipy's nnls keeps
            every parameter >= 0. If the iteration has not settled within max_iterations,
            the likelihood is maximised directly from where it stopped (a safeguard).
            A fixed part (e.g. a scatter or offset held at a given value) is part of the
            model but not fitted.
    Input:  columns    - list of model columns (e.g. decay shape, constant background)
            counts     - raw photons per bin
            fixed_part - optional array added to the model, not fitted (default: none)
    Output: params (one per column, >= 0) and model (expected photons per bin)
    Used by: fit_amplitude_and_background and fitting.fit_sample.
    """
    design = np.column_stack(columns)
    fixed = np.zeros_like(counts, dtype=float) if fixed_part is None else fixed_part
    target = counts - fixed                                    # what the fitted columns must explain
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))          # first guess: variance ≈ counts
    params = nnls(design * weights[:, None], target * weights)[0]
    converged = False
    for _ in range(max_iterations):
        model = np.maximum(design @ params + fixed, 1e-12)
        weights = 1.0 / np.sqrt(model)                         # Poisson: variance = expected counts
        new_params = nnls(design * weights[:, None], target * weights)[0]
        converged = np.allclose(new_params, params, rtol=1e-10, atol=1e-12)
        params = new_params
        if converged:
            break
    if not converged:
        params = _maximise_likelihood_directly(design, counts, params, fixed)
    return params, design @ params + fixed


def _maximise_likelihood_directly(design, counts, start, fixed):
    """
    Safeguard for fit_linear_parameters: minimise the deviance with a general optimiser.

    Why:    the reweighting iteration nearly always settles in a few steps, but it is not
            guaranteed to. Starting from where it stopped, L-BFGS-B with bounds >= 0 and the
            exact gradient finishes the job; the better of the two answers is kept.
    Input:  design - matrix of model columns; counts - photons per bin; start - parameters;
            fixed - part of the model that is not fitted
    Output: parameters (>= 0)
    """
    def deviance_and_gradient(params):
        model = np.maximum(design @ params + fixed, 1e-12)
        gradient = 2.0 * design.T @ (1.0 - counts / model)
        return poisson_deviance(counts, model), gradient

    result = minimize(deviance_and_gradient, start, jac=True, method="L-BFGS-B",
                      bounds=[(0.0, None)] * design.shape[1])
    better = result.fun < poisson_deviance(counts, design @ start + fixed)
    return result.x if better else start


def fit_amplitude_and_background(shape, counts):
    """
    Maximum-likelihood amplitude A and background b for one decay shape.

    Input:  shape  - output of decay_shape
            counts - raw photons per bin (window), not background-subtracted
    Output: amplitude A (counts), background b (counts per bin), model (photons per bin)
    Used by: profile_deviance and the final fit.
    """
    (amplitude, background), model = fit_linear_parameters([shape, np.ones_like(shape)], counts)
    return amplitude, background, model


def profile_deviance(tau_ns, time_ns, counts, irf):
    """
    Deviance at a trial τ, with A and b at their best values for that τ.

    Why:    this "profile" makes the fit a function of τ alone, which fit_single_lifetime
            minimises and errors.profile_error scans.
    Input:  tau_ns (ns), time_ns, counts (raw), irf (unit area)
    Output: D (dimensionless)
    """
    _, _, model = fit_amplitude_and_background(decay_shape(time_ns, irf, tau_ns), counts)
    return poisson_deviance(counts, model)


def fit_single_lifetime(time_ns, counts, irf, n_coarse=40):
    """
    Best single lifetime: the τ that minimises the profile deviance.

    Why:    two stages. A coarse log grid over every plausible lifetime (one bin up to the
            window length) finds the right valley; a bounded scalar minimisation then
            refines τ between the two grid neighbours of the lowest point.
    Input:  time_ns, counts (raw), irf (unit area); n_coarse - points of the coarse grid
    Output: tau_ns (ns) and the deviance at the minimum
    Used by: the λ-m scan (every pair) and therefore the final result.
    """
    coarse_taus = np.geomspace(time_ns[1] - time_ns[0], time_ns[-1] - time_ns[0], n_coarse)
    coarse_deviances = [profile_deviance(tau, time_ns, counts, irf) for tau in coarse_taus]
    k = int(np.argmin(coarse_deviances))
    bracket = (coarse_taus[max(k - 1, 0)], coarse_taus[min(k + 1, n_coarse - 1)])
    result = minimize_scalar(profile_deviance, bounds=bracket, method="bounded",
                             args=(time_ns, counts, irf), options={"xatol": 1e-6})
    return float(result.x), float(result.fun)


def place_irf(irf, start_bin, n_bins, shift_bins=0.0):
    """
    The calibration IRF on a sample's full histogram, at its absolute position, shifted.

    Why:    calibration and sample are different acquisitions, so the timing can drift by a
            fraction of a bin. The shift is applied as a Fourier phase shift: exact for
            smooth signals, and its wrap-around matches the physics because the histogram
            spans one laser period. A positive shift moves the IRF to later times; the area
            stays 1.
    Input:  irf        - unit-area calibration IRF
            start_bin  - absolute bin of irf[0] in the histogram
            n_bins     - number of bins of the histogram
            shift_bins - shift in bins (may be fractional)
    Output: IRF on the full histogram (n_bins values)
    Used by: fitting.fit_sample, and the irf_full_period of every result and calibration.
    """
    placed = np.zeros(n_bins)
    length = min(irf.size, n_bins - start_bin)
    placed[start_bin:start_bin + length] = irf[:length]
    if shift_bins == 0:
        return placed
    frequencies = np.fft.rfftfreq(n_bins)                        # cycles per bin
    phase = np.exp(-2j * np.pi * frequencies * shift_bins)
    return np.fft.irfft(np.fft.rfft(placed) * phase, n=n_bins)


def exponential_component(irf_full, time_ns, tau_ns):
    """
    One exponential component on the full histogram: IRF ⊛ exp(-t/τ), unit amplitude.

    Input:  irf_full - IRF on the full histogram (from place_irf); time_ns (ns); tau_ns (ns)
    Output: one value per bin
    Used by: fitting.fit_sample.
    """
    exponential = np.exp(-(time_ns - time_ns[0]) / tau_ns)
    return np.convolve(irf_full, exponential)[: time_ns.size]
