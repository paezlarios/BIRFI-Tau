"""Measured data: an ATTO655 calibration predicts the lifetime of an independent ATTO655 series."""
from pathlib import Path

import pytest

from tcspc_lifetime import Model, Parameter, SampleResult, build_calibration, read_decay, run_batch

MEASURED = Path(__file__).parents[1] / "examples" / "data" / "measured"


@pytest.fixture(scope="module")
def atto655_calibration():
    return build_calibration([read_decay(p) for p in sorted((MEASURED / "calibration_atto655").glob("*.csv"))])


def test_calibration_dye_lifetime(atto655_calibration):
    c = atto655_calibration
    assert c.n_repetitions == 6
    assert len(c.accepted) >= 5
    assert 1.80 < c.tau_reference_ns < 1.95


def test_independent_series_matches_calibration(atto655_calibration):
    c = atto655_calibration
    model = Model(
        n_components=1,
        lifetimes_ns=[Parameter(value=None, fixed=False, lower=0.1, upper=10.0)],
        shift_bins=Parameter(value=0.0, fixed=False, lower=-3.0, upper=3.0),
        scatter=Parameter(value=0.0, fixed=True),
        offset=Parameter(value=None, fixed=False),
        fit_start_ns=1.0,
        fit_end_ns=20,
    )
    results = run_batch([read_decay(p) for p in sorted((MEASURED / "atto655").glob("*.csv"))], c, model)
    assert len(results) == 6 and all(isinstance(r, SampleResult) for r in results)
    for r in results:
        assert abs(r.value("tau_1_ns") / c.tau_reference_ns - 1) < 0.01, r.decay_name
        assert r.deviance_per_dof < 2.0, r.decay_name
