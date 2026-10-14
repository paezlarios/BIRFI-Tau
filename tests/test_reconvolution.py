"""The linear parameters are the Poisson maximum-likelihood solution."""
import numpy as np

from tcspc_lifetime.core.reconvolution import decay_shape, fit_linear_parameters


def test_iterated_least_squares_solves_the_likelihood_equations(reference_result):
    r = reference_result
    shape = decay_shape(r.time_ns, r.irf, r.tau_ns)
    (amplitude, background), model = fit_linear_parameters([shape, np.ones_like(shape)], r.counts)
    score_amplitude = np.sum(shape * (r.counts / model - 1))      # ∂ℓ/∂A
    score_background = np.sum(r.counts / model - 1)               # ∂ℓ/∂b
    assert abs(score_amplitude) < 1e-6 * np.sum(shape)
    assert abs(score_background) < 1e-6 * r.counts.size
