# BIRFI in Python

Implementation of **blind instrument response function identification** from
Gómez-Sánchez et al., *Biophysical Reports* **4**, 100155 (2024),
[doi:10.1016/j.bpr.2024.100155](https://doi.org/10.1016/j.bpr.2024.100155),
following both the letter and the supporting information — including
transcriptions of the published MATLAB `birfi_ls` and `birfi_dec`.

```python
import numpy as np
from birfi import birfi

t, y = np.loadtxt("decay.txt", unpack=True)   # ns, counts
res = birfi(t, y, method="nnls")

res.irf     # recovered IRF, unit area, on t[:len(irf)] - t[0]
res.t_cut   # cutting point m = end of the IRF support
res.fwhm    # IRF width, for a sanity check against the instrument spec
res.tau     # tail lifetime — reported as a diagnostic, not used by default
```

## The mathematics (SI Eqs. S1–S6)

With `x(t) = A·exp(-t/τ)` for `t ≥ 0` and an IRF of finite support `[0, m]`, the
convolution splits into three domains:

| | | |
|---|---|---|
| Domain 1 | `t < 0` | `y(t) = 0` |
| Domain 2 | `0 ≤ t < m` | `y(t) = A·e^{-t/τ} · ∫₀ᵗ e^{s/τ}·IRF(s) ds` |
| Domain 3 | `t ≥ m` | `y(t) = A·e^{-t/τ} · ∫₀ᵐ e^{s/τ}·IRF(s) ds` |

In Domain 3 the integral is a **constant** `B`, so `y(t) = B·x(t)`. Past the end
of the IRF support the measured decay *is* the underlying decay, merely scaled.
Two consequences drive the algorithm:

1. The point where `y` becomes exponential — the cutting point — is the end of
   the IRF support, so it hands you `m` for free.
2. The tail of `y`, shifted to the origin, *is* `x` up to that scalar. So the
   blind problem "find `h` and `x` from `y`" collapses to the well-posed system
   identification problem "find `h` from `y` and `x`".

Differentiating (Eq. S6) leaves Domain 3 proportional to the original
exponential while distorting Domain 2, which sharpens the frontier.

## Reading the published code

The SI listing is the clearest statement of what the method actually does:

```matlab
function [irf]=birfi_dec(decay,irf_size)
decay_cut=decay(1,irf_size:end);
irf=deconv(decay,decay_cut);
```

**No lifetime is ever estimated.** `x` is the measured tail, verbatim, and the
whole identification is one polynomial long division. `birfi_ls` is the same
system solved by least squares: unwinding the Hankel indexing gives
`HankelM(i,j) = decay_cut(i-j+1)`, i.e. `conv(decay_cut, irf) = decay`. Both are
transcribed here and verified to agree to ~1e-11, as the SI claims, and to
recover the IRF exactly (rel. err ~1e-11) from noise-free data.

`x_source="tail"` is therefore the default. `x_source="model"` is an option that
substitutes a noiseless `exp(-t/τ)` fitted to the tail — see below.

## Pipeline

| Step | Function | Notes |
|---|---|---|
| Background | `estimate_background` | flat level from the pre-pulse bins |
| Smoothing | `whittaker_smooth` | Eilers (2003), sparse 2nd-difference penalty |
| Cutting point | `find_cutting_point` | `derivative_min` (SI) or `tail_extrap` |
| Input signal | `build_input_signal` | the measured tail, or a fitted exponential |
| IRF | `deconvolve_irf` | `lstsq` / `nnls` / `longdiv` |
| Validation | `reconvolution_fit` | reproduces the Fig. 4 residual comparison |

`method="nnls"` (nonnegative IRF) is recommended over plain `lstsq`; it cuts the
error by ~20% at 2% noise at no cost. `"longdiv"` is `birfi_dec` itself — exact,
and the most noise-sensitive.

## Cutting-point detection

The SI's criterion is the **minimum of dy/dt**, treated explicitly as "a
graphical rather than absolute mathematical criterion". I first implemented a
different detector (fit the clean far tail, walk backwards to the departure
point) on the assumption it would be steadier. It is not:

| noise, Eq. S7 | 0.25% | 0.5% | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|---|
| `derivative_min` (SI) | 60 | 60 | 60 | 60 | 60 | 60 |
| `tail_extrap` (mine) | 69 | 68 | 69 | 85 | 111 | 118 |

(bins; true support 52). The SI criterion is essentially noise-invariant and
overshoots slightly, which is the safe direction — surplus IRF samples come out
near zero, whereas undershooting truncates and distorts. `derivative_min` is the
default. Its Savitzky–Golay window barely matters for accuracy but does shift
the detected point, so it defaults to a narrow 1.5% of the trace.

## The smoothing parameter

This matters more than the paper's brief mention suggests. Too little and the
noise-amplifying deconvolution rings, too much and the IRF is artificially
broadened:

| λ (Whittaker) | rel. IRF error, ~2.8% noise |
|---|---|
| 0 | 0.37 |
| 0.5 | 0.13 |
| **2** | **0.084** |
| 20 | 0.12 |
| 100 | 0.17 |

`lam_deconv="auto"` (the default) picks λ by a relative discrepancy principle:
reconvolve each candidate IRF, compare against the *unsmoothed* data, take the
largest λ whose χ² is still within 10% of the best achievable. It lands on the
knee and recovers the hand-tuned optimum.

The underlying conditioning issue: all IRF information lives in the first `m`
samples, where the system is lower triangular with unit diagonal. Exact without
noise, noise-amplifying with it — hence the paper's insistence on Whittaker
pre-smoothing.

## Results on the paper's synthetic setup

τ = 1 ns, 1498 points at 25 ps, three IRF shapes (Fig. S3), Poisson noise
calibrated to **Eq. S7** (`100·‖y-y_f‖/‖y_f‖`) so that 0.5% and 2% mean what they
mean in the paper. 40 repeats.

| | noise-free | 0.5% | 2% |
|---|---|---|---|
| IRF 1 (1 Gaussian) | 0.001 | 0.019 | 0.047 |
| IRF 2 (2 Gaussians) | 0.003 | 0.026 | 0.063 |
| IRF 3 (3 Gaussians) | 0.003 | 0.033 | 0.076 |

(mean relative IRF error). Noise-free recovery is exact, which is the paper's
"flawless, indicative of a well-posed signal identification problem".

The Fig. 4 comparison reproduces too. Against a deliberately mismatched
"measured" IRF (40 ps early, 15% narrower — a realistic colour shift between
scattering at the laser line and detection in the red):

| | τ with BIRFI IRF | χ²_red | τ with mismatched IRF | χ²_red |
|---|---|---|---|---|
| IRF 1 | 1.0000 | 1.03 | 1.0355 | 108.8 |
| IRF 2 | 1.0005 | 0.98 | 1.0356 | 77.8 |
| IRF 3 | 0.9992 | 1.08 | 1.0342 | 71.1 |

A 40 ps IRF error is a 3.5% lifetime error and hugely structured residuals —
the paper's warning that high residuals can come from a bad IRF rather than an
inadequate decay model, and that adding exponential components to chase them is
the wrong move.

## Measured tail vs fitted exponential

Using the measured tail as `x` is an errors-in-variables problem: its noise
enters the design matrix. Substituting a noiseless `exp(-t/τ)` fitted to that
same tail removes it. This costs nothing in assumptions — Eq. S5 only holds if
`x` is monoexponential, so the method already requires it.

Mean relative IRF error, `x_source="tail"` / `"model"`:

| noise, Eq. S7 | 0.25% | 0.5% | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|---|
| IRF 1 | .013/.012 | .020/.019 | .030/.029 | .044/.043 | .123/.099 | .295/.199 |
| IRF 2 | .014/.014 | .026/.026 | .038/.037 | .063/.063 | .127/.108 | .210/.188 |
| IRF 3 | .020/.021 | .031/.031 | .050/.050 | .083/.080 | .135/.126 | .243/.206 |

Below ~2% they are indistinguishable, so the published choice is right for the
paper's own regime and has the virtue of being model-free in practice. Above
that the fitted-exponential variant pulls ahead, by 20–30% at 5–10% noise.
Worth having if you are working at low photon counts.

## Operating range and failure modes

Tested with a ~0.33 ns FWHM IRF at 0.5% noise:

| τ | 0.20 | 0.35 | 0.50 | 1.0 | 2.0 | 5.0 ns |
|---|---|---|---|---|---|---|
| rel. IRF error | 0.023 | 0.030 | 0.022 | 0.022 | 0.024 | 0.024 |

It holds even where τ is shorter than the IRF width, because the method never
needs to resolve the rising edge — only to find where the tail becomes
exponential.

**The one real failure mode is a decay that is not monoexponential.** A
biexponential (0.7·e^{-t/0.4} + 0.3·e^{-t/2.5}) yields an IRF that is 71% wrong.
No internal diagnostic catches this reliably — the tail genuinely *is*
monoexponential once the fast component has died, so tail-based tests pass. The
tell is the **support length**: the cutting point lands at 3.1 ns instead of
1.2 ns, i.e. the "IRF" is claimed to be nanoseconds long. Compare `res.t_cut`
and `res.fwhm` against what your instrument can physically produce. The SI's own
answer is to unmix first with MCR-Slicing (demonstrated there on ATTO 655 /
665 / 647N mixtures) and run BIRFI on each resolved monoexponential component;
that unmixing step is not implemented here.

### Two bugs real data exposed

Both were invisible on simulated data and both are fixed:

- **`estimate_background` over-subtracted.** It used only the pre-pulse bins,
  which returned 73 counts on a trace whose true far-tail baseline was 26.5 —
  the pre-pulse region carried a small afterpulse bump. That drove the
  subtracted tail *negative*. Both regions are only upper bounds on the dark
  level (the pre-pulse can carry afterpulsing, the far tail still holds signal
  if the decay is incomplete), so it now takes the smaller of the two.
- **`select_smoothing` used a 1e-12 variance floor.** With the negative tail
  above, the far-tail bins contributed chi-squared of ~134 each and swamped the
  statistic; the relative discrepancy rule then accepted λ = 0.71 where 0.07 was
  right, broadening the recovered IRF by 60 ps. It now uses a Poisson floor of 1
  count and ignores bins below 0.1% of the model peak.

Other practical points:

- The analysis window must **start before the pulse rises**; the IRF is
  positioned relative to `t[0]`.
- Subtract or fit the background. A sloping baseline breaks the Domain 3
  exponentiality the whole method rests on.
- Set the cutting point manually (`t_cut=...`) when you know it — the paper
  fixes it at 2.00 ns for the Alexa 647 data rather than detecting it.

## Files

- `birfi.py` — the implementation
- `birfi_demo.py` — reproduces Figs. 1, 2, 4 and the tail/model comparison
- `analyse_ch1.py` — worked example on a real TCSPC decay, with robustness
  sweep, parametric bootstrap and an IRF-free cross-check
- `test_si_equivalence.py` — regression tests against the SI listing and Eq. S7
- `fig1_principle.png`, `fig2_montecarlo.png`, `fig4_residuals.png`,
  `fig5_tail_vs_model.png`

Requires numpy, scipy; matplotlib for the demo.
