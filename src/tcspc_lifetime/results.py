"""
Result objects, convertible to JSON-safe dictionaries for the GUI.

Role in the pipeline: what every analysis returns. Field names carry their units; every
array the GUI may plot is included. Internally all times are in ns.
Depends on: NumPy and the standard library.
"""
import math
from dataclasses import asdict, dataclass, field

import numpy as np


def json_safe(value):
    """
    Convert NumPy types, arrays and non-finite numbers to plain JSON values.

    Why:    the GUI receives results as JSON; NaN and infinity are not valid JSON, so they
            become None.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


@dataclass
class PerDecayResult:
    """
    Result of the per-decay analysis (IRF extracted from the decay itself, λ-m approach).

    Identity and settings
        decay_name, source_sha256, library_version, config
    Window (indices refer to the full histogram of the Decay)
        window_start_bin          first bin of the analysis window
        time_ns, counts           window time axis (ns) and raw counts
        background_median_counts  far-tail median background (counts per bin)
        rise_bins                 10-90 % rise time (bins)
    λ-m scan
        m0_bin                    starting cut, within the window (bins)
        lambda_grid_bins4, m_grid_bin, deviance_per_dof_map, tau_map_ns
        near_best_mask            pairs within the tolerance of the best deviance/dof
        irf_distance_map          each pair's IRF distance from the chosen IRF
    Chosen pair and IRF
        best_lambda_bins4, best_m_bin
        irf                       unit-area IRF on window bins 0..best_m;
                                  absolute position in the histogram: window_start_bin + index
    Lifetime and errors (ns)
        tau_ns, tau_profile_error_ns (Δdeviance = 1), tau_fisher_error_ns,
        tau_systematic_ns (half the spread of τ over near-best pairs)
    Fit
        amplitude_counts, background_fit_counts, model_counts, residuals_sigma,
        deviance, degrees_of_freedom, deviance_per_dof,
        curvature_sum_z2, curvature_blocks
    """
    decay_name: str
    source_sha256: str
    library_version: str
    config: dict
    window_start_bin: int
    time_ns: np.ndarray
    counts: np.ndarray
    background_median_counts: float
    rise_bins: float
    m0_bin: int
    lambda_grid_bins4: np.ndarray
    m_grid_bin: np.ndarray
    deviance_per_dof_map: np.ndarray
    tau_map_ns: np.ndarray
    near_best_mask: np.ndarray
    irf_distance_map: np.ndarray
    best_lambda_bins4: float
    best_m_bin: int
    irf: np.ndarray
    tau_ns: float
    tau_profile_error_ns: float
    tau_fisher_error_ns: float
    tau_systematic_ns: float
    amplitude_counts: float
    background_fit_counts: float
    model_counts: np.ndarray
    residuals_sigma: np.ndarray
    deviance: float
    degrees_of_freedom: int
    deviance_per_dof: float
    curvature_sum_z2: float
    curvature_blocks: int
    flags: list = field(default_factory=list)      # filled from build step 4

    def to_dict(self):
        """Every field as plain JSON-safe values (lists, floats, None for NaN)."""
        return json_safe(asdict(self))


@dataclass
class SampleResult:
    """
    Result of fitting one sample with a calibration IRF (fitting.fit_sample).

    Identity and settings
        decay_name, source_sha256, library_version, calibration_name, model (dict)
    Arrays over the full histogram
        time_ns, counts, model_counts, residuals_sigma
        fit_mask        True for bins inside the fitting interval
    parameters
        name -> {"value", "error", "irf_error", "total_error", "free", "kind", "at_bound"}
        kind "fitted": tau_<i>_ns, amplitude_<i>_counts, shift_bins, scatter_counts,
                       offset_counts_per_bin
        kind "derived": amplitude_<i>_percent, intensity_<i>_percent,
                        tau_mean_amplitude_ns, tau_mean_intensity_ns
        error:       Fisher error from the sample's photons (statistical); None for fixed
                     parameters, parameters at a bound, or a singular Fisher matrix
        irf_error:   jackknife error from the calibration IRF's own noise; None when the
                     calibration has fewer than 2 leave-one-out IRFs
        total_error: √(error² + irf_error²); None when either is missing
    Fit quality (inside the fitting interval)
        deviance, degrees_of_freedom, deviance_per_dof, curvature_sum_z2, curvature_blocks
        correlation_names, correlation   correlation matrix of the free fitted parameters
    """
    decay_name: str
    source_sha256: str
    library_version: str
    calibration_name: str
    model: dict
    time_ns: np.ndarray
    counts: np.ndarray
    model_counts: np.ndarray
    residuals_sigma: np.ndarray
    fit_mask: np.ndarray
    parameters: dict
    deviance: float
    degrees_of_freedom: int
    deviance_per_dof: float
    curvature_sum_z2: float
    curvature_blocks: int
    correlation_names: list
    correlation: np.ndarray
    flags: list = field(default_factory=list)      # filled from build step 4

    def value(self, name):
        """Fitted or derived value of a parameter."""
        return self.parameters[name]["value"]

    def error(self, name):
        """Its statistical (Fisher) error, None if not available."""
        return self.parameters[name]["error"]

    def irf_error(self, name):
        """Its jackknife IRF error, None if not available."""
        return self.parameters[name]["irf_error"]

    def total_error(self, name):
        """Statistical and IRF errors combined, None if either is missing."""
        return self.parameters[name]["total_error"]

    def to_dict(self):
        """Every field as plain JSON-safe values (lists, floats, None for NaN)."""
        return json_safe(asdict(self))


@dataclass
class FailedFit:
    """A sample that could not be fitted; the batch continues without it."""
    decay_name: str
    error_message: str

    def to_dict(self):
        return asdict(self)
