"""
fit_sample: one sample decay fitted with a calibration IRF.

Role in the pipeline: the sample stage of calibration mode (prompt, section 5). The model

    μ(t) = Σ_i A_i (IRF_δ ⊛ exp(-t/τ_i)) + S·IRF_δ + b

is computed over the whole histogram, with the IRF at its absolute position from the
calibration, shifted by δ bins. Only bins inside the fitting interval enter the likelihood.
For every trial of the lifetimes and shift, the amplitudes, scatter and offset are solved
exactly by Poisson maximum likelihood (variable projection). Lifetimes and shift are found
by a coarse grid of lifetimes, then Nelder-Mead in log τ, within the model's ranges.
Errors: the Fisher error from the sample's photons, the jackknife IRF error from refitting
with the calibration's leave-one-out IRFs, and their combination.
Depends on: core, model, results, and the Calibration it is given.
"""
import itertools
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from . import __version__
from .core.diagnostics import block_curvature, pearson_residuals
from .core.errors import central_difference, fisher_covariance
from .core.reconvolution import (exponential_component, fit_linear_parameters, place_irf,
                                 poisson_deviance)
from .model import Model, Parameter
from .results import SampleResult

COARSE_POINTS = 16                    # lifetimes per free component in the coarse search
DEFAULT_SHIFT_RANGE_BINS = (-3.0, 3.0)


def fit_sample(decay, calibration, model=None, irf_error=True):
    """
    Fit one sample with the calibration IRF and the chosen model.

    Input:  decay       - the sample (io.Decay), same binning as the calibration
            calibration - calibration.Calibration
            model       - model.Model (default: one free lifetime, free shift and offset)
            irf_error   - also compute the jackknife IRF error (needs the calibration's
                          leave-one-out IRFs; costs one refit per repetition)
    Output: results.SampleResult
    Used by: batch.run_batch and the GUI.
    """
    model = model if model is not None else Model()
    _check_settings(decay, calibration)
    problem = _SampleProblem(decay, calibration, model)
    result = problem.result(problem.refine(problem.coarse_start()))
    if irf_error and len(calibration.jackknife_irfs) >= 2:
        _add_irf_errors(result, decay, calibration, model)
    return result


def _add_irf_errors(result, decay, calibration, model):
    """
    Jackknife IRF error of every parameter, and the combined error.

    Why:    the Fisher error treats the calibration IRF as exact, but the IRF carries the
            calibration's photon noise, shared by every sample of the batch. Refitting with
            each leave-one-out IRF shows how much each parameter moves; because any two
            leave-one-out IRFs share N - 2 repetitions, their raw spread is far too small,
            and σ²_IRF = ((N - 1)/N) Σ_k (θ_k - θ̄)² restores the variance of the estimate
            from all N. Sample and calibration photons are independent, so the combined
            error is √(σ_stat² + σ_IRF²).
    Input:  result (updated in place), decay, calibration, model
    """
    start = _model_starting_at(model, result)
    refits = {name: [] for name in result.parameters}
    for irf_k in calibration.jackknife_irfs:
        problem = _SampleProblem(decay, replace(calibration, irf=irf_k, jackknife_irfs=[]), start)
        refit = problem.result(problem.refine(problem.coarse_start()))
        for name in refits:
            refits[name].append(refit.value(name))
    n = len(calibration.jackknife_irfs)
    for name, entry in result.parameters.items():
        if entry["kind"] == "fitted" and not entry["free"]:
            continue                                    # fixed: no error of any kind
        values = np.array(refits[name])
        entry["irf_error"] = float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2)))
        if entry["error"] is not None:
            entry["total_error"] = float(np.hypot(entry["error"], entry["irf_error"]))


def _model_starting_at(model, result):
    """The same model, with the free lifetimes and shift starting at the fitted values."""
    lifetimes = [Parameter(result.value(f"tau_{i + 1}_ns") if not p.fixed else p.value, p.fixed, p.lower, p.upper)
                 for i, p in enumerate(model.lifetimes_ns)]
    shift = model.shift_bins
    return replace(model, lifetimes_ns=lifetimes,
                   shift_bins=Parameter(result.value("shift_bins"), shift.fixed, shift.lower, shift.upper))


def _check_settings(decay, calibration):
    """Refuse a sample whose binning differs from the calibration's (settings_mismatch)."""
    if decay.counts.size != calibration.n_bins:
        raise ValueError(f"settings_mismatch: sample '{decay.name}' has {decay.counts.size} bins, "
                         f"the calibration {calibration.n_bins}")
    if abs(decay.bin_width_ns - calibration.bin_width_ns) > 1e-6 * calibration.bin_width_ns:
        raise ValueError(f"settings_mismatch: sample '{decay.name}' has bins of "
                         f"{decay.bin_width_ns} ns, the calibration {calibration.bin_width_ns} ns")


class _SampleProblem:
    """
    Everything needed to evaluate the model for one sample.

    The searched (nonlinear) parameters are packed in one vector θ for the optimiser: the
    natural logarithm of each free lifetime, in component order, then the shift if it is free.
    """

    def __init__(self, decay, calibration, model):
        self.decay, self.calibration, self.model = decay, calibration, model
        self.time_ns, self.counts = decay.time_ns, decay.counts
        self.in_fit = _fit_mask(decay.time_ns, calibration, model)
        self.free_lifetimes = [i for i, p in enumerate(model.lifetimes_ns) if not p.fixed]
        self.shift_free = not model.shift_bins.fixed
        interval_ns = self.time_ns[self.in_fit][-1] - self.time_ns[self.in_fit][0]
        self.lifetime_ranges = [(p.lower if p.lower is not None else decay.bin_width_ns,
                                 p.upper if p.upper is not None else interval_ns)
                                for p in model.lifetimes_ns]
        shift = model.shift_bins
        self.shift_range = (shift.lower if shift.lower is not None else DEFAULT_SHIFT_RANGE_BINS[0],
                            shift.upper if shift.upper is not None else DEFAULT_SHIFT_RANGE_BINS[1])

    # --- from θ to the model -------------------------------------------------------------
    def unpack(self, theta):
        """θ -> (lifetimes in ns for every component, shift in bins)."""
        lifetimes = [p.value for p in self.model.lifetimes_ns]
        for k, i in enumerate(self.free_lifetimes):
            lifetimes[i] = float(np.exp(theta[k]))
        shift = float(theta[-1]) if self.shift_free else float(self.model.shift_bins.value)
        return lifetimes, shift

    def components(self, lifetimes, shift):
        """The shifted IRF and one unit-amplitude exponential component per lifetime."""
        irf = place_irf(self.calibration.irf, self.calibration.irf_start_bin, self.counts.size, shift)
        return irf, [exponential_component(irf, self.time_ns, tau) for tau in lifetimes]

    def solve(self, lifetimes, shift):
        """
        Best amplitudes, scatter and offset for given lifetimes and shift.

        Output: linear parameters (amplitudes, then scatter and offset if free), the model
                over the whole histogram, the deviance inside the fitting interval, the
                shifted IRF and the components.
        """
        irf, shapes = self.components(lifetimes, shift)
        columns, fixed_part = list(shapes), np.zeros(self.counts.size)
        if self.model.scatter.fixed:
            fixed_part += self.model.scatter.value * irf
        else:
            columns.append(irf)
        if self.model.offset.fixed:
            fixed_part += self.model.offset.value
        else:
            columns.append(np.ones(self.counts.size))
        m = self.in_fit
        params, _ = fit_linear_parameters([c[m] for c in columns], self.counts[m],
                                          fixed_part=fixed_part[m])
        model_counts = np.column_stack(columns) @ params + fixed_part
        return params, model_counts, poisson_deviance(self.counts[m], model_counts[m]), irf, shapes

    def deviance(self, theta):
        return self.solve(*self.unpack(theta))[2]

    # --- the search --------------------------------------------------------------------------
    def coarse_start(self):
        """
        Starting θ. If every free lifetime has a starting value, use those. Otherwise try a
        log grid of COARSE_POINTS lifetimes per free component (over its range), keeping
        combinations in increasing order when there are any, and start from the best one.
        """
        shift_start = [self.model.shift_bins.value or 0.0] if self.shift_free else []
        free = self.free_lifetimes
        if all(self.model.lifetimes_ns[i].value is not None for i in free):
            return np.array([np.log(self.model.lifetimes_ns[i].value) for i in free] + shift_start)
        grids = [np.geomspace(*self.lifetime_ranges[i], COARSE_POINTS) for i in free]
        combinations = list(itertools.product(*grids))
        increasing = [c for c in combinations if all(a < b for a, b in zip(c, c[1:]))]
        best = min(increasing or combinations,
                   key=lambda c: self.deviance(np.array(list(np.log(c)) + shift_start)))
        return np.array(list(np.log(best)) + shift_start)

    def refine(self, theta):
        """Nelder-Mead from the starting θ, within the ranges (log τ steps of 0.1, shift 0.2 bins)."""
        if theta.size == 0:
            return theta
        bounds = [tuple(np.log(self.lifetime_ranges[i])) for i in self.free_lifetimes]
        steps = [0.1] * len(self.free_lifetimes)
        if self.shift_free:
            bounds.append(self.shift_range)
            steps.append(0.2)
        simplex = [theta] + [theta + step * np.eye(theta.size)[k] for k, step in enumerate(steps)]
        simplex = np.clip(simplex, [b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(self.deviance, theta, method="Nelder-Mead", bounds=bounds,
                          options={"initial_simplex": simplex, "xatol": 1e-7, "fatol": 1e-9,
                                   "maxiter": 5000, "maxfev": 10000})
        return result.x

    # --- the result --------------------------------------------------------------------------
    def result(self, theta):
        """Turn the best θ into a SampleResult with errors, derived quantities and statistics."""
        model, n = self.model, self.model.n_components
        lifetimes, shift = self.unpack(theta)
        params, model_counts, deviance, irf, shapes = self.solve(lifetimes, shift)
        amplitudes = params[:n]
        k = n
        scatter = params[k] if not model.scatter.fixed else model.scatter.value
        k += 0 if model.scatter.fixed else 1
        offset = params[k] if not model.offset.fixed else model.offset.value

        # which parameters are free, and which of those sit on a bound
        at_bound = {}
        for i in range(n):
            lo, hi = self.lifetime_ranges[i]
            at_bound[f"tau_{i + 1}_ns"] = (i in self.free_lifetimes and
                                           (np.isclose(lifetimes[i], lo, rtol=1e-6) or np.isclose(lifetimes[i], hi, rtol=1e-6)))
            at_bound[f"amplitude_{i + 1}_counts"] = amplitudes[i] <= 0
        at_bound["shift_bins"] = self.shift_free and (np.isclose(shift, self.shift_range[0], atol=1e-6) or
                                                     np.isclose(shift, self.shift_range[1], atol=1e-6))
        at_bound["scatter_counts"] = (not model.scatter.fixed) and scatter <= 0
        at_bound["offset_counts_per_bin"] = (not model.offset.fixed) and offset <= 0

        # Jacobian of the model inside the fitting interval, one column per free parameter
        m = self.in_fit
        names, columns = [], []
        for i in self.free_lifetimes:
            name = f"tau_{i + 1}_ns"
            if not at_bound[name]:
                names.append(name)
                columns.append(amplitudes[i] * central_difference(
                    lambda tau: exponential_component(irf, self.time_ns, tau), lifetimes[i])[m])
        if self.shift_free and not at_bound["shift_bins"]:
            def model_at_shift(delta):
                irf_d, shapes_d = self.components(lifetimes, delta)
                return sum(a * s for a, s in zip(amplitudes, shapes_d)) + scatter * irf_d + offset
            names.append("shift_bins")
            columns.append(central_difference(model_at_shift, shift)[m])
        for i in range(n):
            if not at_bound[f"amplitude_{i + 1}_counts"]:
                names.append(f"amplitude_{i + 1}_counts")
                columns.append(shapes[i][m])
        if not model.scatter.fixed and not at_bound["scatter_counts"]:
            names.append("scatter_counts")
            columns.append(irf[m])
        if not model.offset.fixed and not at_bound["offset_counts_per_bin"]:
            names.append("offset_counts_per_bin")
            columns.append(np.ones(int(m.sum())))
        try:
            covariance = fisher_covariance(np.column_stack(columns), model_counts[m])
        except np.linalg.LinAlgError:
            covariance = np.full((len(names), len(names)), np.nan)
        errors = dict(zip(names, np.sqrt(np.abs(np.diag(covariance)))))
        scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        correlation = covariance / np.where(scale > 0, scale, np.nan)

        def entry(name, value, free, kind="fitted"):
            return {"value": float(value), "error": errors.get(name), "irf_error": None,
                    "total_error": None, "free": bool(free), "kind": kind,
                    "at_bound": bool(at_bound.get(name, False))}

        parameters = {}
        for i in range(n):
            parameters[f"tau_{i + 1}_ns"] = entry(f"tau_{i + 1}_ns", lifetimes[i], i in self.free_lifetimes)
            parameters[f"amplitude_{i + 1}_counts"] = entry(f"amplitude_{i + 1}_counts", amplitudes[i], True)
        parameters["shift_bins"] = entry("shift_bins", shift, self.shift_free)
        parameters["scatter_counts"] = entry("scatter_counts", scatter, not model.scatter.fixed)
        parameters["offset_counts_per_bin"] = entry("offset_counts_per_bin", offset, not model.offset.fixed)
        parameters.update(_derived_quantities(lifetimes, amplitudes, names, covariance))

        residuals = pearson_residuals(self.counts, model_counts)
        bins_per_block = max(1, int(round(1.0 / self.decay.bin_width_ns)))
        curvature, n_blocks = block_curvature(residuals[m], bins_per_block)
        n_free = len(self.free_lifetimes) + int(self.shift_free) + n + \
            int(not model.scatter.fixed) + int(not model.offset.fixed)
        dof = int(m.sum()) - n_free
        return SampleResult(
            decay_name=self.decay.name, source_sha256=self.decay.source_sha256,
            library_version=__version__, calibration_name=self.calibration.name,
            model=model.to_dict(), time_ns=self.time_ns, counts=self.counts,
            model_counts=model_counts, residuals_sigma=residuals, fit_mask=m,
            irf_full_period=irf,
            parameters=parameters, deviance=float(deviance), degrees_of_freedom=dof,
            deviance_per_dof=float(deviance / dof), curvature_sum_z2=curvature,
            curvature_blocks=n_blocks, correlation_names=names, correlation=correlation)


def _fit_mask(time_ns, calibration, model):
    """True for the bins inside the fitting interval (default: calibration window to the end)."""
    start = model.fit_start_ns if model.fit_start_ns is not None else time_ns[calibration.irf_start_bin]
    end = model.fit_end_ns if model.fit_end_ns is not None else time_ns[-1]
    tolerance = 1e-6 * (time_ns[1] - time_ns[0])
    mask = (time_ns >= start - tolerance) & (time_ns <= end + tolerance)
    if mask.sum() < 20:
        raise ValueError(f"the fitting interval {start}-{end} ns contains only {mask.sum()} bins")
    return mask


def _derived_quantities(lifetimes, amplitudes, names, covariance):
    """
    Amplitude and intensity fractions and the two mean lifetimes, with propagated errors.

    Why:    individual components can be poorly determined while these are solid; the
            delta method σ_q² = gᵀ C g with the full covariance keeps the correlations.
    Input:  lifetimes, amplitudes (all components); names and covariance of the free
            parameters (from the Fisher information)
    Output: dictionary of derived parameter entries
    """
    n = len(lifetimes)
    v = np.array(list(lifetimes) + list(amplitudes), dtype=float)   # (τ_1..τ_n, A_1..A_n)
    labels = [f"tau_{i + 1}_ns" for i in range(n)] + [f"amplitude_{i + 1}_counts" for i in range(n)]
    c_v = np.zeros((2 * n, 2 * n))                                  # covariance of v (0 if not free)
    index = {name: k for k, name in enumerate(names)}
    for a, la in enumerate(labels):
        for b, lb in enumerate(labels):
            if la in index and lb in index:
                c_v[a, b] = covariance[index[la], index[lb]]

    def propagate(function):
        value = function(v)
        gradient = np.array([central_difference(lambda x: function(np.where(np.arange(2 * n) == k, x, v)),
                                                v[k], 1e-6) for k in range(2 * n)])
        variance = gradient @ c_v @ gradient
        error = float(np.sqrt(variance)) if np.isfinite(variance) else None
        return {"value": float(value), "error": error, "irf_error": None, "total_error": None,
                "free": False, "kind": "derived", "at_bound": False}

    tau, amp = slice(0, n), slice(n, 2 * n)
    derived = {}
    for i in range(n):
        derived[f"amplitude_{i + 1}_percent"] = propagate(lambda x, i=i: 100 * x[amp][i] / x[amp].sum())
        derived[f"intensity_{i + 1}_percent"] = propagate(
            lambda x, i=i: 100 * x[amp][i] * x[tau][i] / np.sum(x[amp] * x[tau]))
    derived["tau_mean_amplitude_ns"] = propagate(lambda x: np.sum(x[amp] * x[tau]) / x[amp].sum())
    derived["tau_mean_intensity_ns"] = propagate(
        lambda x: np.sum(x[amp] * x[tau] ** 2) / np.sum(x[amp] * x[tau]))
    return derived
