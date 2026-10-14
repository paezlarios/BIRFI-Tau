"""
Synthetic TCSPC decays with a known IRF and lifetime.

Role in the pipeline: the only way to show the analysis recovers the truth is to know the
truth. Used by the tests, and later for experiment planning and calibrated model tests.
Depends on: NumPy.
"""
import numpy as np


def gaussian(time_ns, centre_ns, fwhm_ns):
    """Gaussian peak with a given centre and full width at half maximum (ns), peak value 1."""
    sigma_ns = fwhm_ns / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((time_ns - centre_ns) / sigma_ns) ** 2)


def two_peak_irf(time_ns):
    """
    The validation IRF: a main peak (FWHM 0.35 ns at 2.0 ns) and a satellite (FWHM 0.30 ns
    at 2.45 ns, 35 % of the main height), normalised to unit area.
    """
    irf = gaussian(time_ns, 2.0, 0.35) + 0.35 * gaussian(time_ns, 2.45, 0.30)
    return irf / irf.sum()


def expected_decay(irf, time_ns, tau_ns, peak_counts, background=0.0):
    """
    Noise-free expected counts: peak_counts x (IRF ⊛ exp(-t/τ)) scaled to its maximum, + b.

    Why:    the same discrete model the fit uses, so a perfect analysis recovers the truth.
    Input:  irf (unit area, on time_ns), time_ns (ns), tau_ns (ns),
            peak_counts (photons in the highest bin above background), background (per bin)
    Output: expected photons per bin
    """
    shape = np.convolve(irf, np.exp(-(time_ns - time_ns[0]) / tau_ns))[: time_ns.size]
    return peak_counts * shape / shape.max() + background


def poisson_decay(expected, seed=None):
    """Counts with Poisson noise around the expected values (reproducible with a seed)."""
    return np.random.default_rng(seed).poisson(expected).astype(float)


def noise_percent(counts, noise_free):
    """
    Noise level in the BIRFI paper's metric (SI Eq. S7): 100 ||y - y_free|| / ||y_free||.
    """
    return 100.0 * np.sqrt(np.sum((counts - noise_free) ** 2) / np.sum(noise_free ** 2))


def peak_counts_for_noise(irf, time_ns, tau_ns, target_percent):
    """
    Peak counts that give a target noise level in the SI's metric (Eq. S7), on average.

    Why:    for Poisson counts with mean S·u, E[Σ (y - y_free)²] = S·Σu, so the scale hitting
            a target E is S = (100/E)² Σu / Σu², exactly. "0.5 %" or "2 %" then mean what
            they mean in the paper.
    Input:  irf, time_ns, tau_ns - as expected_decay; target_percent - noise level (%)
    Output: peak_counts for expected_decay (no background)
    """
    u = expected_decay(irf, time_ns, tau_ns, 1.0)
    return (100.0 / target_percent) ** 2 * u.sum() / np.sum(u ** 2)


def smooth_irf(time_ns, shift_ns=0.0):
    """
    A smooth IRF resembling the reference decay's: main peak at 2.2 ns (FWHM 0.45 ns) and a
    small late bump at 3.0 ns (6 % of the peak, FWHM 0.35 ns); unit area.

    Why:    a real timing drift shifts the true, smooth IRF. Evaluating this function at
            t - shift gives an exact shift that is independent of the fit's own (Fourier)
            shift, so tests do not just check that an operator agrees with itself.
            (Linear interpolation is not a pure shift: it also broadens the IRF.)
    Input:  time_ns - full histogram time axis (ns); shift_ns - delay (ns)
    Output: IRF on time_ns
    """
    irf = gaussian(time_ns, 2.2 + shift_ns, 0.45) + 0.06 * gaussian(time_ns, 3.0 + shift_ns, 0.35)
    return irf / irf.sum()


def expected_sample(irf_full, time_ns, lifetimes_ns, fractions, peak_counts, background=0.0):
    """
    Noise-free multi-exponential sample: Σ_i A_i (IRF ⊛ exp(-t/τ_i)) + b.

    Input:  irf_full    - IRF on the full histogram
            lifetimes_ns, fractions - lifetimes and amplitude fractions (summing to 1)
            peak_counts - photons in the highest bin above background
            background  - photons per bin
    Output: expected photons per bin, and the true amplitudes A_i (counts)
    """
    shapes = [np.convolve(irf_full, np.exp(-(time_ns - time_ns[0]) / tau))[: time_ns.size]
              for tau in lifetimes_ns]
    decay = sum(f * s for f, s in zip(fractions, shapes))
    scale = peak_counts / decay.max()
    return decay * scale + background, np.array(fractions) * scale
