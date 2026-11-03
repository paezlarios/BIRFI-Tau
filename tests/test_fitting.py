"""Sample fits with a known IRF recover known lifetimes, fractions, shift and offset."""
import json

import numpy as np
import pytest

from tcspc_lifetime import Calibration, Decay, Model, Parameter, fit_sample
from tcspc_lifetime.simulate import expected_sample, poisson_decay, smooth_irf, two_peak_irf

TIME_NS = np.arange(250) * 0.1
CALIBRATION = Calibration.from_known_irf(smooth_irf(TIME_NS), 0, TIME_NS)


def within(result, name, truth, n_sigma=3.0):
    return abs(result.value(name) - truth) < n_sigma * result.error(name)


def test_one_exponential():
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [2.5], [1.0], 2e4, 8.0)
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 1)), CALIBRATION, Model(1))
    assert within(r, "tau_1_ns", 2.5) and within(r, "offset_counts_per_bin", 8.0)
    assert within(r, "shift_bins", 0.0)
    assert 0.8 < r.deviance_per_dof < 1.2


def test_two_exponentials():
    expected, amplitudes = expected_sample(smooth_irf(TIME_NS), TIME_NS, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 2)), CALIBRATION, Model(2))
    for name, truth in [("tau_1_ns", 0.8), ("tau_2_ns", 3.2), ("amplitude_1_percent", 60.0)]:
        assert within(r, name, truth), name
    true_mean = np.sum(amplitudes * np.array([0.8, 3.2]) ** 2) / np.sum(amplitudes * np.array([0.8, 3.2]))
    assert within(r, "tau_mean_intensity_ns", true_mean)


def test_exact_subbin_shift_with_narrow_irf():
    """A 0.4-bin delay of a narrow IRF (FWHM 3.5 bins), made by evaluating it at t - 0.04 ns."""
    calibration = Calibration.from_known_irf(two_peak_irf(TIME_NS), 0, TIME_NS)
    expected, _ = expected_sample(two_peak_irf(TIME_NS - 0.04), TIME_NS, [2.5], [1.0], 2e4, 8.0)
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 3)), calibration, Model(1))
    assert within(r, "shift_bins", 0.4) and within(r, "tau_1_ns", 2.5)


def test_fixed_parameter_interval_and_degrees_of_freedom():
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)
    model = Model(2, lifetimes_ns=[Parameter(0.8, fixed=True), Parameter(lower=1.5, upper=8.0)],
                  fit_start_ns=1.5, fit_end_ns=20.0)
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 2)), CALIBRATION, model)
    assert r.value("tau_1_ns") == 0.8 and r.error("tau_1_ns") is None
    assert not r.parameters["tau_1_ns"]["free"]
    assert r.fit_mask.sum() == 186                                   # bins 1.5 ... 20.0 ns
    assert r.degrees_of_freedom == 186 - 5                           # τ2, shift, A1, A2, offset
    assert within(r, "tau_2_ns", 3.2)


def test_fixed_offset_and_free_scatter():
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [2.5], [1.0], 2e4, 8.0)
    expected = expected + 3000 * smooth_irf(TIME_NS)                 # scattered excitation light
    model = Model(1, scatter=Parameter(0.0), offset=Parameter(8.0, fixed=True))
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 4)), CALIBRATION, model)
    assert r.value("offset_counts_per_bin") == 8.0
    assert within(r, "scatter_counts", 3000) and within(r, "tau_1_ns", 2.5)


def test_irf_on_the_full_histogram_with_fixed_shift():
    """Shift held at 0: the calibration IRF at its absolute bins 15..34, zeros elsewhere."""
    compact = smooth_irf(TIME_NS)[15:35]
    calibration = Calibration.from_known_irf(compact, 15, TIME_NS)
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [2.5], [1.0], 2e4, 8.0)
    model = Model(1, shift_bins=Parameter(0.0, fixed=True))
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 6)), calibration, model)
    full = r.irf_full_period
    assert full.size == 250
    assert np.allclose(full[15:35], compact / compact.sum(), rtol=1e-14, atol=0)
    assert not full[:15].any() and not full[35:].any()
    assert len(r.to_dict()["irf_full_period"]) == 250


def test_irf_on_the_full_histogram_is_shifted_as_fitted():
    """The reported IRF is the one the fit used: its centroid moves by the fitted shift, area 1."""
    expected, _ = expected_sample(smooth_irf(TIME_NS, 0.04), TIME_NS, [2.5], [1.0], 2e4, 8.0)
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 7)), CALIBRATION, Model(1))
    full = r.irf_full_period
    bins = np.arange(TIME_NS.size)
    assert r.value("shift_bins") > 0.2                               # a real shift to follow
    assert (bins * full).sum() - (bins * CALIBRATION.irf).sum() == pytest.approx(r.value("shift_bins"), abs=1e-4)
    assert full.sum() == pytest.approx(1.0, abs=1e-12)


def test_refuses_different_binning():
    with pytest.raises(ValueError, match="settings_mismatch"):
        fit_sample(Decay(np.arange(200) * 0.1, np.ones(200)), CALIBRATION, Model(1))


def test_result_is_json_safe():
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [2.5], [1.0], 2e4, 8.0)
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 5)), CALIBRATION, Model(1))
    json.dumps(r.to_dict(), allow_nan=False)


@pytest.mark.slow
def test_error_coverage_two_exponentials():
    """Fraction of fits whose 1σ Fisher error covers the truth, with a known IRF (target 68 %)."""
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)
    hits = {"tau_1_ns": [], "tau_2_ns": [], "amplitude_1_percent": []}
    truths = {"tau_1_ns": 0.8, "tau_2_ns": 3.2, "amplitude_1_percent": 60.0}
    for seed in range(40):
        r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 1000 + seed)), CALIBRATION, Model(2))
        for name in hits:
            hits[name].append(within(r, name, truths[name], n_sigma=1.0))
    coverage = {name: np.mean(h) for name, h in hits.items()}
    print("\n1σ coverage:", {k: f"{100 * v:.0f} %" for k, v in coverage.items()})
    assert all(0.50 <= c <= 0.85 for c in coverage.values())


def test_jackknife_irf_error_and_combined_error():
    """With leave-one-out IRFs, every free parameter gets an IRF error and a combined error."""
    rng = np.random.default_rng(7)
    noisy = [smooth_irf(TIME_NS) * (1 + 0.02 * rng.normal(size=TIME_NS.size)) for _ in range(4)]
    calibration = Calibration.from_known_irf(smooth_irf(TIME_NS), 0, TIME_NS)
    calibration.jackknife_irfs = [irf / irf.sum() for irf in noisy]
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)
    model = Model(2, lifetimes_ns=[Parameter(0.8, fixed=True), Parameter(lower=1.5, upper=8.0)])
    r = fit_sample(Decay(TIME_NS, poisson_decay(expected, 2)), calibration, model)
    for name in ("tau_2_ns", "amplitude_1_percent", "shift_bins", "tau_mean_intensity_ns"):
        assert r.irf_error(name) > 0
        assert r.total_error(name) == pytest.approx(np.hypot(r.error(name), r.irf_error(name)))
    assert r.irf_error("tau_1_ns") is None and r.total_error("tau_1_ns") is None     # fixed
    plain = fit_sample(Decay(TIME_NS, poisson_decay(expected, 2)), calibration, model, irf_error=False)
    assert plain.irf_error("tau_2_ns") is None and plain.value("tau_2_ns") == r.value("tau_2_ns")
