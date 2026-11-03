"""Regression: the per-decay analysis reproduces the reference numbers (prompt, section 8)."""
import json

import numpy as np
import pytest


def test_reference_decay_regression(reference_result):
    """Acceptance: λ and m identical; τ within 1e-4 ns of 1.8682; deviance/dof within 0.001."""
    r = reference_result
    assert r.best_m_bin == 21
    assert r.best_lambda_bins4 == pytest.approx(0.02218, rel=1e-3)
    assert abs(r.tau_ns - 1.8682) < 1e-4
    assert abs(r.deviance_per_dof - 1.125) < 1e-3


def test_reference_decay_details(reference_result):
    r = reference_result
    assert r.time_ns[0] == pytest.approx(1.0)                 # window starts at 1.0 ns
    assert r.rise_bins == pytest.approx(4.25, abs=0.01)
    assert r.m0_bin == 21
    assert r.near_best_mask.sum() == 24
    assert np.nanmax(r.irf_distance_map[r.near_best_mask]) < 0.028   # within 2.7 %
    assert abs(r.tau_profile_error_ns - 0.0018) < 0.0001
    assert abs(r.tau_systematic_ns - 0.0057) < 0.0001


def test_irf_on_the_full_histogram(reference_result):
    """Window bins 0..21 of the IRF sit at histogram bins 10..31; zeros everywhere else."""
    r = reference_result
    full = r.irf_full_period
    assert full.size == 250
    assert np.array_equal(full[10:32], r.irf)
    assert not full[:10].any() and not full[32:].any()
    assert len(r.to_dict()["irf_full_period"]) == 250


def test_result_is_json_safe(reference_result):
    text = json.dumps(reference_result.to_dict(), allow_nan=False)
    assert '"tau_ns"' in text
