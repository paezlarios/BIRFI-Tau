"""
Diagnostics of a fit: residuals, residual curvature, IRF distances.

Role in the pipeline: numbers that tell the user (and later the quality flags) whether a
fit can be trusted.
Depends on: NumPy, SciPy.
"""
import numpy as np
from scipy.stats import norm


def pearson_residuals(counts, model):
    """
    Residuals in units of the Poisson σ: (counts - model)/√model.

    Why:    puts every bin on the same ±1 scale, so structure is visible across the whole
            decay, from the peak (σ ≈ 200 counts) to the tail (σ ≈ 5).
    Input:  counts, model - photons per bin
    Output: residual per bin (dimensionless)
    """
    return (counts - model) / np.sqrt(np.maximum(model, 1e-12))


def block_curvature(residuals, bins_per_block):
    """
    Residual-curvature statistic: Σ z² over blocks of consecutive bins.

    Why:    a misfit bends the residuals over nanoseconds, which a bin-by-bin runs test
            misses. Averaging residuals over a block and multiplying by √(bins per block)
            gives a z-score per block; for a correct model Σ z² is close to the number of
            blocks (χ² with one degree of freedom per block).
    Input:  residuals      - Pearson residuals
            bins_per_block - block length in bins (1 ns by default)
    Output: (sum of z², number of full blocks used)
    """
    n_blocks = residuals.size // bins_per_block
    blocks = residuals[: n_blocks * bins_per_block].reshape(n_blocks, bins_per_block)
    z = blocks.mean(axis=1) * np.sqrt(bins_per_block)
    return float(np.sum(z ** 2)), int(n_blocks)


def irf_distance(irf, reference_irf):
    """
    Relative difference between two IRFs: ||IRF - reference|| / ||reference||.

    Why:    one number for how far an IRF is from the chosen one; IRFs of different length
            are compared with zeros after their last sample (an IRF is zero beyond it).
    Input:  irf, reference_irf - unit-area IRFs, possibly of different lengths
    Output: relative L2 distance (0 = identical)
    """
    length = max(irf.size, reference_irf.size)
    a, b = np.zeros(length), np.zeros(length)
    a[: irf.size], b[: reference_irf.size] = irf, reference_irf
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def repetition_shape_test(counts, other_counts):
    """
    log10 of the p-value that one repetition has the same shape as the others.

    Why:    if the shapes agree, each bin's counts split between this repetition and the
            others in one fixed proportion, with binomial noise. Pearson's χ² of that
            hypothesis has (bins - 1) degrees of freedom. No IRF and no lifetime are needed,
            so the test stays blind and also catches an intruder that is not
            monoexponential (prompt, section 4). Pearson's χ² is used rather than the
            binomial deviance because it stays calibrated in bins with few counts: on clean
            simulated sets at 2,000 counts peak the deviance flagged 1.27 % of repetitions
            at p < 1e-3 and Pearson 0.05 % (0.1 % expected), with the same power.
            The logarithm of p is returned, computed through the Wilson-Hilferty normal
            approximation of χ², z = ((X²/k)^(1/3) - (1 - 2/(9k))) / √(2/(9k)), whose log tail
            stays finite. For a gross mismatch the p-value itself underflows to 0 (and so
            does SciPy's chi2.logsf), and several repetitions tied at p = 0 could not be
            ranked: a sample file slipped into the calibration set then got three clean
            repetitions excluded instead of itself.
    Input:  counts       - photons per bin of one repetition
            other_counts - photons per bin of the sum of the other repetitions
    Output: log10 of the p-value (very negative: probably a different sample)
    Used by: calibration.build_calibration.
    """
    total = counts + other_counts
    keep = total > 0
    y, n = counts[keep], total[keep]
    share = y.sum() / n.sum()
    pearson = np.sum((y - n * share) ** 2 / (n * share * (1.0 - share)))
    dof = keep.sum() - 1
    z = ((pearson / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(2.0 / (9.0 * dof))
    return float(norm.logsf(z) / np.log(10.0))
