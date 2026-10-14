"""Shared fixtures: the reference decay and its per-decay analysis (computed once)."""
from pathlib import Path

import pytest

from tcspc_lifetime import PerDecayConfig, analyse, read_decay

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def reference_decay():
    return read_decay(DATA / "LifeTimeHistogramCh1.csv")


@pytest.fixture(scope="session")
def reference_result(reference_decay):
    """Per-decay analysis with the reference grid (m0 - 3 to m0 + 6), as in section 8."""
    return analyse(reference_decay, PerDecayConfig.reference_grid())
