"""
Statistical errors: from the profile deviance and from the Fisher information.

Role in the pipeline: the error bars of fitted parameters due to the decay's own photon
noise. (The IRF's own noise is added separately in calibration mode, by the jackknife.)
Depends on: NumPy, SciPy.
"""
import numpy as np
from scipy.optimize import brentq


def profile_error(deviance_at, best_value, max_relative_step=0.5):
    """
    One-standard-deviation error of one parameter: where its profile deviance rises by 1.

    Why:    for a single parameter, the rise of the deviance above its minimum follows a
            chi-squared distribution with one degree of freedom (Wilks' theorem), so a rise
            of 1 marks 1 σ. The other parameters are re-optimised at every trial value, so
            their correlations are included. Each crossing is found with brentq.
    Input:  deviance_at - function: parameter value -> profile deviance
            best_value  - the fitted value (the deviance minimum)
            max_relative_step - give up if no crossing within this fraction of the value
    Output: error (same unit as the parameter), the average of the two sides;
            nan if the deviance does not rise by 1 within the search range
    Used by: per_decay.analyse.
    """
    deviance_min = deviance_at(best_value)

    def excess(value):                     # deviance above the minimum, minus 1: zero at 1 σ
        return deviance_at(value) - deviance_min - 1.0

    def crossing(direction):               # step away until the excess turns positive
        step = 1e-3 * best_value
        while excess(best_value + direction * step) < 0 and step < max_relative_step * best_value:
            step *= 2.0
        if excess(best_value + direction * step) < 0:
            return np.nan
        return brentq(excess, *sorted((best_value, best_value + direction * step)))

    return 0.5 * (crossing(+1) - crossing(-1))


def fisher_covariance(jacobian, model):
    """
    Covariance matrix of the fitted parameters from the Poisson Fisher information.

    Why:    the information F = Jᵀ diag(1/μ) J measures how sharply the likelihood peaks:
            J[i, j] = ∂μ_i/∂θ_j says how much bin i's expected counts move when parameter j
            moves, and 1/μ_i weights each bin by the inverse of its Poisson variance. Its
            inverse is the covariance (Cramér-Rao limit, reached by maximum likelihood at
            high counts). The full matrix is inverted, so the compensation between
            parameters (τ against A and b) is included.
    Input:  jacobian - array (bins x free parameters); model - expected counts per bin
    Output: covariance matrix (free parameters x free parameters);
            errors are the square roots of its diagonal
    Used by: per_decay.analyse.
    """
    information = jacobian.T @ (jacobian / np.maximum(model, 1e-12)[:, None])
    return np.linalg.inv(information)


def central_difference(function, value, relative_step=1e-4):
    """
    Derivative of an array-valued function of one parameter, by central differences.

    Input:  function - parameter -> array; value - where to differentiate
    Output: array, d function / d parameter
    Used by: the Jacobian columns of nonlinear parameters (lifetimes, later the shift).
    """
    step = relative_step * abs(value) if value != 0 else relative_step
    return (function(value + step) - function(value - step)) / (2.0 * step)
