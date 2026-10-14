"""
Reading decays: the Decay object and one reader per file format.

Role in the pipeline: turns a file into a validated Decay (time in ns, counts), keeping the
whole histogram - not only a fitting window - plus the file's SHA-256 for reproducibility.
Readers sit behind one function, read_decay, which picks the reader from the file suffix,
so new formats are added by registering one function in READERS.
Depends on: NumPy and the standard library; no other module of this package.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .csv_reader import read_csv_columns


@dataclass
class Decay:
    """
    One TCSPC histogram.

    time_ns        start time of each bin (ns), uniform spacing
    counts         photons per bin (the full histogram, never cropped)
    name           label chosen by the caller (for example the file name)
    source_sha256  SHA-256 of the file it was read from ("" if built in memory)
    metadata       anything the GUI wants to attach (acquisition order, channel, ...)
    """
    time_ns: np.ndarray
    counts: np.ndarray
    name: str = ""
    source_sha256: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time_ns = np.asarray(self.time_ns, dtype=float)
        self.counts = np.asarray(self.counts, dtype=float)
        _validate(self.time_ns, self.counts, self.name)

    @property
    def bin_width_ns(self):
        """Width of one bin (ns)."""
        return float(np.median(np.diff(self.time_ns)))


def _validate(time_ns, counts, name):
    """
    Refuse histograms the analysis cannot use, with a message that says why.

    Why:    a wrong file should fail at once and clearly, not deep inside a fit.
    Input:  time_ns, counts - the two columns; name - for the error message
    Output: None; raises ValueError on the first problem found
    Used by: Decay.__post_init__.
    """
    label = f"decay '{name}'" if name else "decay"
    if time_ns.ndim != 1 or counts.ndim != 1 or time_ns.size != counts.size:
        raise ValueError(f"{label}: time and counts must be two 1-D columns of equal length")
    if counts.size < 20:
        raise ValueError(f"{label}: only {counts.size} bins; at least 20 are needed")
    if not (np.all(np.isfinite(time_ns)) and np.all(np.isfinite(counts))):
        raise ValueError(f"{label}: contains NaN or infinite values")
    if np.any(counts < 0):
        raise ValueError(f"{label}: contains negative counts")
    steps = np.diff(time_ns)
    if np.any(steps <= 0):
        raise ValueError(f"{label}: time must increase from bin to bin")
    if np.ptp(steps) > 1e-3 * np.median(steps):
        raise ValueError(f"{label}: bins are not uniformly spaced")
    bin_width = np.median(steps)
    if not 1e-3 <= bin_width <= 2.0:
        raise ValueError(f"{label}: bin width {bin_width:g} ns is implausible; "
                         "is the time column in picoseconds? Use time_unit='ps'.")


READERS = {".csv": read_csv_columns}          # file suffix -> function(path, time_unit)


def read_decay(path, time_unit="ns", name=None, metadata=None):
    """
    Read a decay from a file, choosing the reader from the file suffix.

    Why:    the single entry point for files; the GUI never needs to know the format.
    Input:  path      - file to read
            time_unit - unit of the time column in the file: "ns" or "ps"
            name      - label for the decay (default: the file name)
            metadata  - optional dictionary supplied by the GUI
    Output: Decay
    Used by: the GUI and the tests.
    """
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"no reader for '{path.suffix}' files; known: {sorted(READERS)}")
    time_ns, counts = reader(path, time_unit)
    return Decay(time_ns=time_ns, counts=counts, name=name or path.name,
                 source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 metadata=dict(metadata or {}))
