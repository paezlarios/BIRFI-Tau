"""The Whittaker smoother and the λ <-> cutoff relation."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tcspc_lifetime.core.smoothing import cutoff_for_lambda, lambda_for_cutoff, whittaker_smooth


def test_package_equals_direct_solve(reference_decay):
    """Acceptance: whittaker-eilers solves (I + λ D2ᵀD2) z = y to within 1e-10 of the peak."""
    y = reference_decay.counts[10:] - 27.0
    second_difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(y.size - 2, y.size))
    for lam in (0.0222, lambda_for_cutoff(3 * 4.25), 1e3):
        direct = spsolve((sparse.identity(y.size) + lam * second_difference.T @ second_difference).tocsc(), y)
        assert np.abs(whittaker_smooth(y, lam) - direct).max() < 1e-10 * np.abs(y).max()


def test_lambda_zero_does_not_smooth():
    y = np.random.default_rng(0).normal(size=50)
    assert np.array_equal(whittaker_smooth(y, 0.0), y)


def test_cutoff_relation_is_exact():
    for cutoff in (2.5, 4.0, 12.75, 40.0):
        lam = lambda_for_cutoff(cutoff)
        assert cutoff_for_lambda(lam) == np.float64(cutoff) or abs(cutoff_for_lambda(lam) - cutoff) < 1e-9
        omega = 2 * np.pi / cutoff                      # the gain is exactly 1/2 at the cutoff
        assert abs(1 / (1 + 16 * lam * np.sin(omega / 2) ** 4) - 0.5) < 1e-12
