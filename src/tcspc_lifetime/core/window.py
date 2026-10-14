"""
Background, analysis window and rise time.

Role in the pipeline: the first step on every decay. The background is subtracted before
the deconvolution, the window fixes the time origin of the IRF, and the rise time sets the
scale of both smoothing choices.
Depends on: NumPy.
"""
import numpy as np


def estimate_background(counts, tail_fraction=0.10, min_bins=10):
    """
    Flat background level: dark counts and stray light, present in every bin.

    Why:    the deconvolution must see only the fluorescence decay, so the background is
            subtracted first. The far tail is used because the fluorescence has died away
            there; the median ignores the occasional outlier bin.
    Input:  counts        - photons per bin (the whole histogram)
            tail_fraction - fraction of the last bins used (default: last 10 %)
            min_bins      - never fewer bins than this
    Output: background (photons per bin)
    Used by: find_window_start, measure_rise_time, and the background-subtracted decay.
    """
    n_tail_bins = max(min_bins, int(tail_fraction * counts.size))
    return float(np.median(counts[-n_tail_bins:]))


def find_window_start(time_ns, counts, background, margin_ns=0.8, rise_threshold=0.02):
    """
    Index of the first bin of the analysis window: shortly before the pulse.

    Why:    BIRFI positions the IRF relative to the window's first bin and counts its
            length from there (IRF length = m + 1 bins). Starting just before the rise keeps
            the IRF short while keeping its whole leading edge inside the window. The index
            is kept, so the IRF's absolute position in the histogram is index + IRF bin.
    Input:  time_ns, counts - the whole histogram
            background      - photons per bin (from estimate_background)
            margin_ns       - how far before the rise the window starts (ns)
            rise_threshold  - the rise is the first bin above this fraction of the peak
                              height above background
    Output: first_bin - index of the window's first bin in the histogram
    Used by: per_decay.analyse (and later the calibration).
    """
    height_above_background = counts - background
    first_rising_bin = int(np.argmax(height_above_background >
                                     rise_threshold * height_above_background.max()))
    bin_width_ns = np.median(np.diff(time_ns))
    return max(0, first_rising_bin - int(round(margin_ns / bin_width_ns)))


def measure_rise_time(counts, background):
    """
    10-90 % rise time of the decay's leading edge, in bins.

    Why:    the rising edge is shaped by the IRF, so its duration is a proxy for the IRF
            width. Measured on the raw counts: the edge has many counts per bin, and
            smoothing it would only make it look slower.
    Input:  counts     - photons per bin (analysis window)
            background - photons per bin
    Output: rise time in bins (fractional, by linear interpolation)
    Used by: cut.find_initial_cut and smoothing.make_lambda_grid.
    """
    height = counts - background
    peak_bin = int(np.argmax(height))
    edge = height[: peak_bin + 1] / height[peak_bin]     # leading edge, 1 at the peak

    def crossing_bin(level):
        # last bin below 'level' before the peak, then linear interpolation into the next bin
        below = np.flatnonzero(edge < level)[-1]
        return below + (level - edge[below]) / (edge[below + 1] - edge[below])

    return crossing_bin(0.9) - crossing_bin(0.1)
