"""
Per-decay mode: the IRF is extracted blindly from the decay being fitted (λ-m approach).

Role in the pipeline: for experiments without a calibration. Also the reference pipeline
whose numbers the library must reproduce (prompt, section 8).
Steps: background and window -> rise time -> starting cut -> λ-m scan (smooth, cut,
deconvolve, normalise, fit) -> best pair -> errors and diagnostics.
Depends on: core, config, results.
"""
import numpy as np

from . import __version__
from .config import PerDecayConfig
from .core.cut import find_initial_cut, make_m_grid
from .core.deconvolution import deconvolve_irf, normalise_irf
from .core.diagnostics import block_curvature, irf_distance, pearson_residuals
from .core.errors import central_difference, fisher_covariance, profile_error
from .core.reconvolution import (decay_shape, fit_amplitude_and_background, fit_single_lifetime,
                                 place_irf, poisson_deviance, profile_deviance)
from .core.smoothing import make_lambda_grid, whittaker_smooth
from .core.window import estimate_background, find_window_start, measure_rise_time
from .results import PerDecayResult

N_FITTED_PARAMETERS = 3                     # τ, amplitude, background


def analyse(decay, config=None):
    """
    The whole per-decay analysis of one decay.

    Why:    one call from a Decay to a lifetime with its errors, the IRF and everything the
            GUI may show.
    Input:  decay  - a Decay (from io.read_decay)
            config - PerDecayConfig (default settings if None)
    Output: PerDecayResult
    Used by: the GUI (directly, or through the batch functions of later build steps).
    """
    config = config or PerDecayConfig()

    # window and scales
    background = estimate_background(decay.counts, config.background_tail_fraction)
    start = find_window_start(decay.time_ns, decay.counts, background,
                              config.window_margin_ns, config.rise_threshold)
    time_ns, counts = decay.time_ns[start:], decay.counts[start:]
    y = counts - background                          # input of the deconvolution
    rise_bins = measure_rise_time(counts, background)
    m0 = find_initial_cut(y, rise_bins, config.cut_smoothing_factor)

    # λ-m scan
    lambda_grid = make_lambda_grid(rise_bins, config.lambda_top_factor,
                                   config.lambda_values, config.lambda_decades)
    m_grid = make_m_grid(m0, counts.size, config.m_before, config.m_after, config.min_tail_bins)
    deviance_per_dof_map, tau_map, irfs = _scan(time_ns, counts, y, lambda_grid, m_grid, config)

    # best pair
    i_best, j_best = np.unravel_index(np.argmin(deviance_per_dof_map), deviance_per_dof_map.shape)
    near_best = deviance_per_dof_map <= (1.0 + config.near_best_tolerance) * deviance_per_dof_map[i_best, j_best]
    best_m = int(m_grid[j_best])
    irf = irfs[i_best][j_best]
    tau_ns = float(tau_map[i_best, j_best])
    distance_map = np.array([[irf_distance(other, irf) if other is not None else np.nan
                              for other in row] for row in irfs])

    # final fit, errors, diagnostics
    shape = decay_shape(time_ns, irf, tau_ns)
    amplitude, background_fit, model = fit_amplitude_and_background(shape, counts)
    profile_err = profile_error(lambda tau: profile_deviance(tau, time_ns, counts, irf), tau_ns)
    fisher_err = _lifetime_fisher_error(time_ns, irf, tau_ns, amplitude, background_fit, model)
    residuals = pearson_residuals(counts, model)
    bins_per_block = max(1, int(round(config.curvature_block_ns / decay.bin_width_ns)))
    curvature, n_blocks = block_curvature(residuals, bins_per_block)
    deviance = poisson_deviance(counts, model)
    dof = counts.size - N_FITTED_PARAMETERS

    return PerDecayResult(
        decay_name=decay.name, source_sha256=decay.source_sha256, library_version=__version__,
        config=config.to_dict(), window_start_bin=int(start), time_ns=time_ns, counts=counts,
        background_median_counts=background, rise_bins=float(rise_bins), m0_bin=int(m0),
        lambda_grid_bins4=lambda_grid, m_grid_bin=m_grid, deviance_per_dof_map=deviance_per_dof_map,
        tau_map_ns=tau_map, near_best_mask=near_best, irf_distance_map=distance_map,
        best_lambda_bins4=float(lambda_grid[i_best]), best_m_bin=best_m, irf=irf,
        irf_full_period=place_irf(irf, start, decay.counts.size),
        tau_ns=tau_ns, tau_profile_error_ns=float(profile_err), tau_fisher_error_ns=fisher_err,
        tau_systematic_ns=float(0.5 * np.ptp(tau_map[near_best])),
        amplitude_counts=float(amplitude), background_fit_counts=float(background_fit),
        model_counts=model, residuals_sigma=residuals, deviance=float(deviance),
        degrees_of_freedom=int(dof), deviance_per_dof=float(deviance / dof),
        curvature_sum_z2=curvature, curvature_blocks=n_blocks)


def _scan(time_ns, counts, y, lambda_grid, m_grid, config):
    """
    Fit every (λ, m) pair and record how well its IRF explains the raw counts.

    Why:    neither λ nor m is fixed by the method, so each pair is judged by the same
            Poisson deviance used to fit τ. Too little smoothing gives a ragged IRF, too much
            a broadened one; too early a cut truncates the IRF, too late adds noisy samples -
            all show up as a worse fit. Each λ smooths once; every cut reuses that trace.
    Input:  time_ns, counts (window, raw); y (counts above background); the two grids
    Output: deviance_per_dof[i_lambda, i_m] (inf where a pair failed),
            tau_map[i_lambda, i_m] (ns; nan where a pair failed),
            irfs[i_lambda][i_m] (unit-area IRF, or None where a pair failed)
    """
    dof = counts.size - N_FITTED_PARAMETERS
    deviance_per_dof = np.full((lambda_grid.size, m_grid.size), np.inf)
    tau_map = np.full((lambda_grid.size, m_grid.size), np.nan)
    irfs = [[None] * m_grid.size for _ in lambda_grid]
    for i, lam in enumerate(lambda_grid):
        y_smoothed = whittaker_smooth(y, lam)
        for j, m in enumerate(m_grid):
            irf = normalise_irf(deconvolve_irf(y_smoothed, m), config.clip_negative_irf)
            if irf is None:
                continue
            tau, deviance = fit_single_lifetime(time_ns, counts, irf, config.lifetime_grid_points)
            deviance_per_dof[i, j], tau_map[i, j], irfs[i][j] = deviance / dof, tau, irf
    return deviance_per_dof, tau_map, irfs


def _lifetime_fisher_error(time_ns, irf, tau_ns, amplitude, background, model):
    """
    Fisher error of τ for the model A·(IRF ⊛ exp(-t/τ)) + b.

    Why:    gives the same 1σ as the profile deviance for one lifetime, and generalises to
            several parameters at once (sample fits, later build steps).
    Input:  window time axis, IRF, fitted τ, A, b and model
    Output: error of τ (ns)
    """
    shape = decay_shape(time_ns, irf, tau_ns)
    d_model_d_tau = amplitude * central_difference(lambda tau: decay_shape(time_ns, irf, tau), tau_ns)
    columns = [d_model_d_tau, shape]                  # ∂μ/∂τ, ∂μ/∂A
    if background > 0:                                # a parameter at its bound is not free
        columns.append(np.ones_like(shape))           # ∂μ/∂b
    covariance = fisher_covariance(np.column_stack(columns), model)
    return float(np.sqrt(covariance[0, 0]))
