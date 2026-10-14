"""Calibration: excluding a different sample, refusing what cannot be calibrated, saving."""
import numpy as np
import pytest

from tcspc_lifetime import Calibration, CalibrationError, Decay, build_calibration
from tcspc_lifetime.simulate import expected_sample, poisson_decay, smooth_irf

TIME_NS = np.arange(250) * 0.1


def repetition(tau_ns, seed, name):
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [tau_ns], [1.0], 1e4, 5.0)
    return Decay(TIME_NS, poisson_decay(expected, seed), name=name)


@pytest.fixture(scope="module")
def calibration_with_intruder():
    """Five repetitions of a 1.87 ns dye; the fourth is a different sample (2.2 ns)."""
    return build_calibration([repetition(2.2 if k == 3 else 1.87, 300 + k, f"rep{k + 1}") for k in range(5)])


def test_excludes_the_different_sample(calibration_with_intruder):
    c = calibration_with_intruder
    assert [e["name"] for e in c.excluded] == ["rep4"]
    assert c.accepted == ["rep1", "rep2", "rep3", "rep5"]
    assert all(c.shape_test_p_values[name] > 1e-3 for name in c.accepted)


def test_reference_lifetime_and_irf(calibration_with_intruder):
    c = calibration_with_intruder
    assert abs(c.tau_reference_ns - 1.87) < 3 * c.tau_reference_error_ns + 0.006   # + systematic allowance
    truth = smooth_irf(TIME_NS)
    found = np.zeros_like(truth)
    found[c.irf_start_bin:c.irf_start_bin + c.irf.size] = c.irf
    # one realisation (4 x 10^4 counts peak): the 5 % criterion applies to the mean of many
    assert np.linalg.norm(found - truth) / np.linalg.norm(truth) < 0.08


def test_two_repetitions_that_disagree():
    with pytest.raises(CalibrationError, match="two repetitions differ"):
        build_calibration([repetition(1.87, 1, "a"), repetition(2.2, 2, "b")])


def test_too_few_left_after_exclusion():
    with pytest.raises(CalibrationError, match="only 2 repetitions remain"):
        build_calibration([repetition(1.87, 1, "a"), repetition(1.87, 2, "b"), repetition(2.2, 3, "c")])


def test_different_binning():
    other = Decay(np.arange(200) * 0.1, np.ones(200), name="short")
    with pytest.raises(CalibrationError, match="settings_mismatch"):
        build_calibration([repetition(1.87, 1, "a"), other])


def test_save_and_load(calibration_with_intruder, tmp_path):
    path = tmp_path / "calibration.json"
    calibration_with_intruder.save(path)
    loaded = Calibration.load(path)
    assert np.array_equal(loaded.irf, calibration_with_intruder.irf)
    assert loaded.irf_start_bin == calibration_with_intruder.irf_start_bin
    assert loaded.excluded == calibration_with_intruder.excluded


def test_keeps_one_leave_one_out_irf_per_accepted_repetition(calibration_with_intruder):
    c = calibration_with_intruder
    assert len(c.jackknife_irfs) == len(c.accepted) == 4
    for irf in c.jackknife_irfs:
        assert irf.size == c.irf.size and abs(irf.sum() - 1) < 1e-12
        assert 0 < np.linalg.norm(irf - c.irf) / np.linalg.norm(c.irf) < 0.05   # close, not identical


def test_save_and_load_keeps_leave_one_out_irfs(calibration_with_intruder, tmp_path):
    path = tmp_path / "calibration.json"
    calibration_with_intruder.save(path)
    loaded = Calibration.load(path)
    assert len(loaded.jackknife_irfs) == 4
    assert all(np.array_equal(a, b) for a, b in zip(loaded.jackknife_irfs, calibration_with_intruder.jackknife_irfs))


def test_chooses_lambda_and_m_by_leaving_one_repetition_out(calibration_with_intruder):
    c = calibration_with_intruder
    assert c.selection == "cross_validated"
    assert c.selection_score_map.shape == (c.lambda_grid_bins4.size, c.m_grid_bin.size)
    i, j = np.unravel_index(np.argmin(c.selection_score_map), c.selection_score_map.shape)
    assert c.best_lambda_bins4 == c.lambda_grid_bins4[i] and c.best_m_bin == c.m_grid_bin[j]


def test_single_repetition_uses_the_decay_itself():
    c = build_calibration([repetition(1.87, 1, "only")])
    assert c.selection == "same_data" and c.jackknife_irfs == []


def test_a_sample_slipped_into_the_calibration_set_is_the_one_excluded():
    """Regression: gross mismatches tie at p = 0; ranking must use log p, or clean ones go."""
    clean = [repetition(1.87, 400 + k, f"rep{k + 1}") for k in range(4)]
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)
    intruder = Decay(TIME_NS, poisson_decay(expected, 9), name="sample")
    c = build_calibration(clean + [intruder])
    assert [e["name"] for e in c.excluded] == ["sample"]
    assert c.excluded[0]["log10_p_value"] < -100
    assert c.accepted == ["rep1", "rep2", "rep3", "rep4"]
