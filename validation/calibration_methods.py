"""
Validation of the two proposed calibration methods (prompt, sections 4 and 10), on simulated
experiments with a known IRF and known lifetimes. One realisation = 5 calibration repetitions
(τ = 1.87 ns, 1e4 counts peak, background 5) + 4 two-component samples (0.8/3.2 ns, 60:40,
2e4 peak, background 8). Methods for the calibration IRF:
  same_clip  current: λ-m chosen on the summed decay itself, negative IRF samples clipped
  cv_free    proposed: λ-m chosen by leaving one repetition out, no clipping
  cv_clip    leaving one repetition out, with clipping (separates the two changes)
For each method: IRF error, calibration lifetime, sample parameters, Fisher error and the
jackknife IRF error (leave-one-out IRFs at the chosen λ, m).
"""
import json, sys, time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize_scalar

from tcspc_lifetime import Calibration, Decay, Model, Parameter, PerDecayConfig, analyse, fit_sample
from tcspc_lifetime.core.cut import find_initial_cut, make_m_grid
from tcspc_lifetime.core.deconvolution import deconvolve_irf, normalise_irf
from tcspc_lifetime.core.reconvolution import fit_single_lifetime, profile_deviance
from tcspc_lifetime.core.smoothing import make_lambda_grid, whittaker_smooth
from tcspc_lifetime.core.window import estimate_background, find_window_start, measure_rise_time
from tcspc_lifetime.simulate import expected_sample, poisson_decay, smooth_irf

T = np.arange(250) * 0.1
TRUE_IRF = smooth_irf(T)
N, CFG = 5, PerDecayConfig()
TRUTH = {"tau_1_ns": 0.8, "tau_2_ns": 3.2, "amplitude_1_percent": 60.0}


def irf_from(counts, start, lam, m, clip):
    y = counts[start:] - estimate_background(counts)
    return normalise_irf(deconvolve_irf(whittaker_smooth(y, lam), m), clip)


def cv_choice(reps, clip):
    total = sum(reps)
    bg = estimate_background(total)
    start = find_window_start(T, total, bg, CFG.window_margin_ns, CFG.rise_threshold)
    tw, cw = T[start:], total[start:]
    rise = measure_rise_time(cw, bg)
    m0 = find_initial_cut(cw - bg, rise)
    lams, ms = make_lambda_grid(rise), make_m_grid(m0, cw.size, CFG.m_before, CFG.m_after)
    tau0, _ = fit_single_lifetime(tw, cw, irf_from(total, start, lams[len(lams) // 2], m0, True))
    others = [total - r for r in reps]
    score = np.full((lams.size, ms.size), np.inf)
    for i, lam in enumerate(lams):
        ys = [whittaker_smooth(o[start:] - estimate_background(o), lam) for o in others]
        for j, m in enumerate(ms):
            total_dev = 0.0
            for k in range(N):
                irf = normalise_irf(deconvolve_irf(ys[k], m), clip)
                if irf is None:
                    total_dev = np.inf; break
                total_dev += minimize_scalar(profile_deviance, bounds=(0.5 * tau0, 2 * tau0), method="bounded",
                                             args=(tw, reps[k][start:], irf), options={"xatol": 1e-5}).fun
            score[i, j] = total_dev
    i, j = np.unravel_index(np.argmin(score), score.shape)
    edge = (j in (0, ms.size - 1)) or i == lams.size - 1
    return start, lams[i], int(ms[j]), bool(edge)


def realisation(seed):
    rng_seeds = 10_000 + 100 * seed
    reps = [poisson_decay(expected_sample(TRUE_IRF, T, [1.87], [1.0], 1e4, 5.0)[0], rng_seeds + k) for k in range(N)]
    samples = [Decay(T, poisson_decay(expected_sample(TRUE_IRF, T, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)[0], rng_seeds + 50 + s),
                     name=f"s{s}") for s in range(4)]
    total = sum(reps)
    out = {"seed": seed}
    t_start = time.time()
    a = analyse(Decay(T, total))
    choices = {"same_clip": (a.window_start_bin, a.best_lambda_bins4, a.best_m_bin, True, False)}
    for name, clip in (("cv_free", False), ("cv_clip", True)):
        start, lam, m, edge = cv_choice(reps, clip)
        choices[name] = (start, lam, m, clip, edge)
    for method, (start, lam, m, clip, edge) in choices.items():
        irf = irf_from(total, start, lam, m, clip)
        placed = np.zeros(250); placed[start:start + irf.size] = irf
        tau_ref, _ = fit_single_lifetime(T[start:], total[start:], irf)
        jack = [irf_from(total - reps[k], start, lam, m, clip) for k in range(N)]
        cal = Calibration.from_known_irf(irf, start, T)
        rows = []
        for sample in samples:
            base = Model(2, lifetimes_ns=[Parameter(lower=0.2, upper=1.8), Parameter(lower=1.8, upper=8.0)], fit_start_ns=1.0)
            r = fit_sample(sample, cal, base)
            start_model = Model(2, lifetimes_ns=[Parameter(r.value("tau_1_ns"), lower=0.2, upper=1.8),
                                                 Parameter(r.value("tau_2_ns"), lower=1.8, upper=8.0)],
                                shift_bins=Parameter(r.value("shift_bins"), lower=-3, upper=3), fit_start_ns=1.0)
            jk = {p: [] for p in TRUTH}
            for irf_k in jack:
                rk = fit_sample(sample, Calibration.from_known_irf(irf_k, start, T), start_model)
                for p in TRUTH:
                    jk[p].append(rk.value(p))
            rows.append({p: {"value": r.value(p), "stat": r.error(p),
                             "irf": float(np.sqrt((N - 1) / N * np.sum((np.array(jk[p]) - np.mean(jk[p])) ** 2)))}
                         for p in TRUTH} | {"dev_dof": r.deviance_per_dof})
        out[method] = {"lambda": float(lam), "m": m, "edge": edge,
                       "irf_error": float(np.linalg.norm(placed - TRUE_IRF) / np.linalg.norm(TRUE_IRF)),
                       "tau_ref": float(tau_ref), "samples": rows}
    out["seconds"] = time.time() - t_start
    return out


if __name__ == "__main__":
    first, count, workers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    with Pool(workers) as pool, open(f"calibration_methods_results.jsonl", "w") as handle:
        for result in pool.imap_unordered(realisation, range(first, first + count)):
            handle.write(json.dumps(result) + "\n"); handle.flush()
