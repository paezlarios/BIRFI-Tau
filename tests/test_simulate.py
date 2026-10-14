"""The simulator reproduces the noise metric of the BIRFI paper (SI Eq. S7)."""
import numpy as np
import pytest

from tcspc_lifetime.simulate import (expected_decay, noise_percent, peak_counts_for_noise,
                                     poisson_decay, two_peak_irf)


@pytest.mark.parametrize("target", [0.5, 2.0])
def test_noise_calibration(target):
    """Acceptance: mean realised noise within 5 % of the target over 100 realisations."""
    time_ns = np.arange(250) * 0.1
    irf = two_peak_irf(time_ns)
    expected = expected_decay(irf, time_ns, 1.2, peak_counts_for_noise(irf, time_ns, 1.2, target))
    realised = [noise_percent(poisson_decay(expected, seed=s), expected) for s in range(100)]
    assert abs(np.mean(realised) / target - 1) < 0.05
