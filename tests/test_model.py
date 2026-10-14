"""The model object refuses inconsistent settings."""
import pytest

from tcspc_lifetime import Model, Parameter


def test_defaults():
    model = Model(2)
    assert len(model.lifetimes_ns) == 2 and not model.lifetimes_ns[0].fixed
    assert model.scatter.fixed and model.scatter.value == 0       # as in SPCImage
    assert not model.offset.fixed and not model.shift_bins.fixed


@pytest.mark.parametrize("make, message", [
    (lambda: Model(4), "1, 2 or 3"),
    (lambda: Model(2, lifetimes_ns=[Parameter()]), "need 2"),
    (lambda: Parameter(fixed=True), "needs a value"),
    (lambda: Parameter(lower=3, upper=1), "below"),
    (lambda: Parameter(5.0, lower=1, upper=3), "outside"),
    (lambda: Model(1, offset=Parameter(upper=10)), "not supported"),
    (lambda: Model(1, fit_start_ns=5, fit_end_ns=2), "before"),
    (lambda: Model(1, lifetimes_ns=[Parameter(-1.0)]), "positive"),
])
def test_refuses(make, message):
    with pytest.raises(ValueError, match=message):
        make()
