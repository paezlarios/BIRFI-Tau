"""Recovery of a known lifetime and IRF on simulated decays (slow: run with -m slow)."""
import numpy as np
import pytest

from tcspc_lifetime import Decay, analyse
from tcspc_lifetime.simulate import expected_decay, poisson_decay, two_peak_irf


@pytest.mark.slow
def test_twenty_realisations():
    """Acceptance: τ bias within 3 standard errors of zero; mean IRF error below 5 %."""
    time_ns = np.arange(250) * 0.1
    irf_true = two_peak_irf(time_ns)
    expected = expected_decay(irf_true, time_ns, 1.2, 5e4, background=25.0)
    taus, irf_errors = [], []
    for seed in range(20):
        result = analyse(Decay(time_ns, poisson_decay(expected, seed=seed)))
        taus.append(result.tau_ns)
        truth = irf_true[result.window_start_bin:]
        truth = truth / truth.sum()
        found = np.zeros_like(truth)
        found[: result.irf.size] = result.irf
        irf_errors.append(np.linalg.norm(found - truth) / np.linalg.norm(truth))
    bias = np.mean(taus) - 1.2
    standard_error = np.std(taus, ddof=1) / np.sqrt(len(taus))
    print(f"\nbias {bias*1e3:+.2f} ± {standard_error*1e3:.2f} ps, scatter {np.std(taus, ddof=1)*1e3:.2f} ps, "
          f"mean IRF error {100*np.mean(irf_errors):.2f} %")
    assert abs(bias) < 3 * standard_error
    assert np.mean(irf_errors) < 0.05
