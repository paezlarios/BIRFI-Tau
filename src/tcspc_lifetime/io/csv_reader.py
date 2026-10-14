"""
Reader for two-column CSV files: time and counts, '#' lines are a header.

Role in the pipeline: the only file format of version 1. Registered in io.READERS.
Depends on: NumPy.
"""
import numpy as np

_TIME_UNITS_IN_NS = {"ns": 1.0, "ps": 1e-3}


def read_csv_columns(path, time_unit="ns"):
    """
    Read the time and counts columns of a TCSPC histogram saved as CSV.

    Why:    the format exported by the acquisition software: comma-separated, time first.
    Input:  path      - the CSV file
            time_unit - unit of the time column: "ns" or "ps"
    Output: time_ns (ns), counts (photons per bin)
    Used by: io.read_decay.
    """
    if time_unit not in _TIME_UNITS_IN_NS:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS_IN_NS)}, not '{time_unit}'")
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns (time, counts), found {data.shape[1]}")
    return data[:, 0] * _TIME_UNITS_IN_NS[time_unit], data[:, 1]
