"""
Calibration: the IRF from N repetitions of a monoexponential calibration dye.

Role in the pipeline: the first stage of calibration mode (prompt, section 4).
    1. Check that all repetitions have the same binning (settings_mismatch).
    2. Check that every repetition is the same sample: a model-free shape test of each
       repetition against the sum of the others; the worst failing one is excluded and the
       rest re-tested, until all pass (p >= 1e-3). With fewer than 3 left, or 2 that
       disagree, the calibration fails.
    3. Sum the accepted repetitions and choose λ and m by cross-validation across
       repetitions: for every (λ, m) pair, the IRF is built from all accepted repetitions but
       one and scored by the Poisson deviance of a fit of the one left out, rotating over all
       of them. The IRF is never judged on its own photons. The final IRF is extracted from
       the full sum at the chosen pair. Validated on 16 simulated experiments: 0.27 ± 0.05 %
       lower IRF error than choosing on the sum itself (paired, better in 81 %), no
       significant bias. Negative IRF samples are still clipped: without clipping, the same
       selection gave a 3.7σ lifetime bias. With a single repetition, λ and m are chosen on
       the decay itself (per-decay pipeline).
    4. Keep the N leave-one-out IRFs (each from the sum of all accepted repetitions but one,
       at the same window, λ and m), so every sample fit can measure how much its parameters
       depend on the calibration's own noise (the jackknife IRF error, fitting.fit_sample).
Depends on: io, core, per_decay, results.
"""
import json
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.optimize import minimize_scalar

from . import __version__
from .config import PerDecayConfig
from .core.deconvolution import deconvolve_irf, normalise_irf
from .core.diagnostics import repetition_shape_test
from .core.errors import profile_error
from .core.reconvolution import fit_single_lifetime, place_irf, profile_deviance
from .core.smoothing import whittaker_smooth
from .core.window import estimate_background
from .io import Decay
from .per_decay import analyse
from .results import json_safe

SHAPE_TEST_THRESHOLD_LOG10 = -3.0           # exclude below p = 1e-3


class CalibrationError(ValueError):
    """The calibration cannot be built (calibration_failed or settings_mismatch)."""


@dataclass
class Calibration:
    """
    The calibration IRF and how it was obtained.

    name                  label (e.g. "sum of 4 calibration repetitions")
    time_ns               time axis of the histograms (ns)
    irf                   unit-area IRF
    irf_start_bin         absolute bin of irf[0] in the histogram
    irf_full_period       (property) the IRF on every bin of the histogram, zeros elsewhere;
                          in to_dict() for the GUI, left out by save(), ignored by load()
    best_lambda_bins4, best_m_bin   chosen smoothing and cut
    tau_reference_ns, tau_reference_error_ns   lifetime of the summed calibration decay
    n_repetitions         number of repetitions given (N)
    accepted              names of the repetitions summed
    excluded              [{"name", "p_value", "log10_p_value"}] repetitions excluded as a
                          different sample (p_value may underflow to 0; log10_p_value does not)
    shape_test_p_values   name -> p-value of the final shape test
    centroid_offsets_bins name -> timing of each repetition relative to the median (drift)
    summed_counts         the summed histogram
    source_sha256         SHA-256 of each repetition's file
    library_version, config
    jackknife_irfs        the leave-one-out IRFs (one per accepted repetition; empty if
                          fewer than 2 were accepted)
    selection             "cross_validated" or "same_data" (how λ and m were chosen)
    lambda_grid_bins4, m_grid_bin, selection_score_map
                          the grids tried and the cross-validation score of each pair
                          (sum of held-out deviances; empty for "same_data")
    analysis              per-decay analysis of the summed decay (not saved to file)
    """
    name: str
    time_ns: np.ndarray
    irf: np.ndarray
    irf_start_bin: int
    best_lambda_bins4: float = float("nan")
    best_m_bin: int = -1
    tau_reference_ns: float = float("nan")
    tau_reference_error_ns: float = float("nan")
    n_repetitions: int = 0
    accepted: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    shape_test_p_values: dict = field(default_factory=dict)
    centroid_offsets_bins: dict = field(default_factory=dict)
    summed_counts: np.ndarray = None
    source_sha256: list = field(default_factory=list)
    library_version: str = __version__
    config: dict = field(default_factory=dict)
    jackknife_irfs: list = field(default_factory=list)
    selection: str = "same_data"
    lambda_grid_bins4: np.ndarray = None
    m_grid_bin: np.ndarray = None
    selection_score_map: np.ndarray = None
    analysis: object = None

    @property
    def bin_width_ns(self):
        return float(np.median(np.diff(self.time_ns)))

    @property
    def n_bins(self):
        return int(self.time_ns.size)

    @property
    def irf_full_period(self):
        return place_irf(self.irf, self.irf_start_bin, self.n_bins)

    @classmethod
    def from_known_irf(cls, irf, irf_start_bin, time_ns, name="known IRF"):
        """A calibration from an IRF that is already known (simulations, or a measured IRF)."""
        irf = np.asarray(irf, dtype=float)
        return cls(name=name, time_ns=np.asarray(time_ns, dtype=float), irf=irf / irf.sum(),
                   irf_start_bin=int(irf_start_bin))

    def to_dict(self):
        """Every field except the full analysis, plus irf_full_period, as JSON-safe values."""
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "analysis"}
        return json_safe({**data, "irf_full_period": self.irf_full_period})

    def save(self, path):
        """Write the calibration to a JSON file, so a batch can be refitted later."""
        data = self.to_dict()
        del data["irf_full_period"]                      # derived; version 0.3.1 cannot read it
        with open(path, "w") as handle:
            json.dump(data, handle)

    @classmethod
    def load(cls, path):
        """Read a calibration written by save()."""
        with open(path) as handle:
            data = json.load(handle)
        data.pop("irf_full_period", None)                # derived from irf and irf_start_bin
        for key in ("time_ns", "irf", "summed_counts", "lambda_grid_bins4", "m_grid_bin", "selection_score_map"):
            data[key] = np.array(data[key], dtype=float) if data.get(key) is not None else None
        data["jackknife_irfs"] = [np.array(irf, dtype=float) for irf in data.get("jackknife_irfs", [])]
        return cls(**data)


def build_calibration(decays, config=None):
    """
    Build the calibration from N repetitions of the calibration dye.

    Input:  decays - list of io.Decay, the N repetitions (N >= 1; 3 or more recommended)
            config - PerDecayConfig for the IRF extraction (default settings if None)
    Output: Calibration
    Raises: CalibrationError when the binning differs, when two repetitions disagree, or
            when fewer than 3 remain after excluding mismatched ones
    Used by: the GUI, before fitting the samples.
    """
    decays = list(decays)
    if not decays:
        raise CalibrationError("no calibration decays given")
    names = [d.name or f"repetition {k + 1}" for k, d in enumerate(decays)]
    _check_same_settings(decays, names)
    accepted, excluded, p_values = _exclude_mismatched(decays, names)
    summed = Decay(time_ns=decays[0].time_ns, counts=sum(decays[k].counts for k in accepted),
                   name=f"sum of {len(accepted)} calibration repetitions")
    config = config or PerDecayConfig()
    analysis = analyse(summed, config)                    # window, grids, and the same-data choice
    start = analysis.window_start_bin
    lam, m, selection, score_map = analysis.best_lambda_bins4, analysis.best_m_bin, "same_data", None
    if len(accepted) >= 2:
        lam, m, score_map = _cross_validated_choice([decays[k].counts for k in accepted], summed.counts,
                                                    analysis, config)
        selection = "cross_validated"
    irf = irf_at(summed.counts, start, lam, m, config)
    tau_reference = analysis.tau_ns
    tau_error = analysis.tau_profile_error_ns
    if selection == "cross_validated":
        time_window, counts_window = summed.time_ns[start:], summed.counts[start:]
        tau_reference, _ = fit_single_lifetime(time_window, counts_window, irf, config.lifetime_grid_points)
        tau_error = profile_error(lambda tau: profile_deviance(tau, time_window, counts_window, irf), tau_reference)
    jackknife_irfs = []
    if len(accepted) >= 2:
        jackknife_irfs = [irf_at(summed.counts - decays[k].counts, start, lam, m, config) for k in accepted]
    return Calibration(
        name=summed.name, time_ns=summed.time_ns, irf=irf, irf_start_bin=start,
        best_lambda_bins4=float(lam), best_m_bin=int(m), tau_reference_ns=float(tau_reference),
        tau_reference_error_ns=float(tau_error), n_repetitions=len(decays),
        accepted=[names[k] for k in accepted], excluded=excluded, shape_test_p_values=p_values,
        centroid_offsets_bins=_centroid_offsets(decays, names, accepted),
        summed_counts=summed.counts, source_sha256=[d.source_sha256 for d in decays],
        config=config.to_dict(), jackknife_irfs=jackknife_irfs, selection=selection,
        lambda_grid_bins4=analysis.lambda_grid_bins4, m_grid_bin=analysis.m_grid_bin,
        selection_score_map=score_map, analysis=analysis)


def _cross_validated_choice(repetitions, summed_counts, analysis, config):
    """
    Choose λ and m by leaving one repetition out.

    Why:    scored on the decay it was extracted from, an IRF is rewarded for absorbing that
            decay's noise. Here each candidate IRF is built from all repetitions but one and
            judged by how well it fits the one left out, whose noise it never saw. Each
            held-out fit is a single lifetime with its own amplitude and background, searched
            within a factor 2 of the summed decay's lifetime (the dye is the same in every
            repetition). One smoothing per λ and left-out repetition; every cut reuses it.
    Input:  repetitions   - counts of the accepted repetitions (full histograms)
            summed_counts - their sum
            analysis      - per-decay analysis of the sum (window, grids, starting lifetime)
            config        - PerDecayConfig
    Output: λ, m, and the score map (sum of held-out deviances, λ x m)
    """
    start = analysis.window_start_bin
    time_window = analysis.time_ns
    lambda_grid, m_grid = analysis.lambda_grid_bins4, analysis.m_grid_bin
    tau_bounds = (0.5 * analysis.tau_ns, 2.0 * analysis.tau_ns)
    left_out_sums = [summed_counts - counts for counts in repetitions]
    score = np.full((lambda_grid.size, m_grid.size), np.inf)
    for i, lam in enumerate(lambda_grid):
        smoothed = [whittaker_smooth(others[start:] - estimate_background(others, config.background_tail_fraction), lam)
                    for others in left_out_sums]
        for j, m in enumerate(m_grid):
            total = 0.0
            for held_out, y_smoothed in zip(repetitions, smoothed):
                irf = normalise_irf(deconvolve_irf(y_smoothed, m), config.clip_negative_irf)
                if irf is None:
                    total = np.inf
                    break
                total += minimize_scalar(profile_deviance, bounds=tau_bounds, method="bounded",
                                         args=(time_window, held_out[start:], irf),
                                         options={"xatol": 1e-5}).fun
            score[i, j] = total
    i, j = np.unravel_index(np.argmin(score), score.shape)
    return float(lambda_grid[i]), int(m_grid[j]), score


def irf_at(counts, window_start_bin, lam, m, config):
    """
    The IRF of a histogram at a given window start, λ and cut, as in the per-decay pipeline.

    Why:    the leave-one-out IRFs must differ from the calibration IRF only by the photons
            left out, so they reuse its window, λ and m; only the background is re-estimated,
            because a sum of fewer repetitions has less of it.
    Input:  counts - histogram (full); window_start_bin, lam, m - from the calibration;
            config - PerDecayConfig (background fraction, clipping)
    Output: unit-area IRF
    Used by: build_calibration.
    """
    background = estimate_background(counts, config.background_tail_fraction)
    y_smoothed = whittaker_smooth(counts[window_start_bin:] - background, lam)
    return normalise_irf(deconvolve_irf(y_smoothed, m), config.clip_negative_irf)


def _check_same_settings(decays, names):
    """All repetitions must share the number of bins and the bin width (settings_mismatch)."""
    first = decays[0]
    for decay, name in zip(decays[1:], names[1:]):
        if (decay.counts.size != first.counts.size or
                abs(decay.bin_width_ns - first.bin_width_ns) > 1e-6 * first.bin_width_ns):
            raise CalibrationError(f"settings_mismatch: '{name}' has {decay.counts.size} bins of "
                                   f"{decay.bin_width_ns} ns; '{names[0]}' has {first.counts.size} "
                                   f"bins of {first.bin_width_ns} ns")


def _exclude_mismatched(decays, names):
    """
    Remove repetitions that are a different sample, one at a time, worst first.

    Why:    a wrong sample contaminates the sum that every clean repetition is compared
            with, so after each exclusion the rest are re-tested among themselves. The worst
            is the one with the lowest log10 p (p itself can tie at 0 for gross mismatches).
    Output: accepted indices, excluded [{"name", "p_value", "log10_p_value"}],
            final p-value per name
    """
    accepted, excluded, p_values = list(range(len(decays))), [], {}
    if len(decays) == 1:
        return accepted, excluded, {names[0]: None}
    while True:
        log_p = {k: repetition_shape_test(decays[k].counts,
                                          sum(decays[j].counts for j in accepted if j != k))
                 for k in accepted}
        worst = min(log_p, key=log_p.get)
        if log_p[worst] >= SHAPE_TEST_THRESHOLD_LOG10:
            p_values.update({names[k]: 10.0 ** log_p[k] for k in accepted})
            return accepted, excluded, p_values
        if len(accepted) == 2:
            raise CalibrationError(
                f"calibration_failed: the two repetitions differ (log10 p = {log_p[worst]:.1f}); with "
                "two it cannot be told which one is a different sample. Measure at least three.")
        excluded.append({"name": names[worst], "p_value": 10.0 ** log_p[worst],
                         "log10_p_value": log_p[worst]})
        p_values[names[worst]] = 10.0 ** log_p[worst]
        accepted.remove(worst)
        if len(accepted) < 3:
            raise CalibrationError(
                f"calibration_failed: only {len(accepted)} repetitions remain after excluding "
                f"{[e['name'] for e in excluded]}; at least 3 are needed")


def _centroid_offsets(decays, names, accepted):
    """
    Timing of each repetition's peak region relative to the median of the accepted ones.

    Why:    a drift of the laser or electronics between repetitions shifts the IRF; this is
            reported (repetition_drift) but does not exclude a repetition.
    Output: name -> offset in bins
    """
    centroids = []
    for decay in decays:
        height = decay.counts - estimate_background(decay.counts)
        weights = np.where(height > 0.1 * height.max(), height, 0.0)
        centroids.append(np.sum(np.arange(height.size) * weights) / weights.sum())
    reference = np.median([centroids[k] for k in accepted])
    return {name: float(c - reference) for name, c in zip(names, centroids)}
