"""Reading decays and refusing unusable ones."""
import numpy as np
import pytest

from tcspc_lifetime import Decay, read_decay


def test_reads_reference_decay(reference_decay):
    assert reference_decay.counts.size == 250
    assert reference_decay.bin_width_ns == pytest.approx(0.1)
    assert len(reference_decay.source_sha256) == 64


def test_picosecond_time_column(tmp_path):
    path = tmp_path / "ps.csv"
    np.savetxt(path, np.column_stack([np.arange(100) * 100.0, np.ones(100)]), delimiter=",")
    assert read_decay(path, time_unit="ps").bin_width_ns == pytest.approx(0.1)
    with pytest.raises(ValueError, match="picoseconds"):
        read_decay(path)                     # read as ns, a 100 ns bin is implausible


@pytest.mark.parametrize("counts, message", [
    (np.r_[np.ones(49), -1.0], "negative"),
    (np.r_[np.ones(49), np.nan], "NaN"),
    (np.ones(10), "at least 20"),
])
def test_refuses_bad_counts(counts, message):
    with pytest.raises(ValueError, match=message):
        Decay(np.arange(counts.size) * 0.1, counts)


def test_refuses_unknown_format(tmp_path):
    path = tmp_path / "decay.xyz"
    path.write_text("1,2\n")
    with pytest.raises(ValueError, match="no reader"):
        read_decay(path)
