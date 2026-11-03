# LibraryMasterPrompt

Oct 2, 2026 · @F

## 0. Instructions to the implementer

You are implementing version 1 of the Python library specified in this document, as a multi-file package with the layout in section 6. Read the whole document before writing any code.

**Task.** Build version 1 following the build order in section 7, one step at a time. After each step, run its tests and report three things: what was built, the test results against the acceptance criteria in section 8, and any deviation from this specification. Then stop and wait for review before starting the next step.

**Rules.**

- Sections 3–7 are requirements; section 9 explains why they were chosen. Do not change a decision without asking.
- Methods marked *Status: proposed* must not become defaults until their validation test passes (section 10). Build the test first.
- When something is unspecified, choose the simplest option consistent with this document, make it a configuration value, and list it in your step report.
- When an open question in section 10 blocks progress, ask. Where section 10 gives a recommendation, follow it unless told otherwise.
- Readability and clarity come before speed (section 7).

**Files to attach to the session.**

| File | Needed for |
| --- | --- |
| `birfi_lambda_m.ipynb` | reference implementation; the section 8 regression numbers |
| `reconvolution.py` | reference Poisson maximum-likelihood reconvolution |
| `LifeTimeHistogramCh1.csv` | regression data: the reference decay |
| `birfi.py`, `test_si_equivalence.py` | transcriptions of the SI's MATLAB code and their tests |
| `whittaker_explorer.ipynb` | optional: what λ does |
| BIRFI paper and its SI (PDF) | optional: background on the method |

**Glossary.**

- **Reference decay** — `LifeTimeHistogramCh1.csv`: 250 bins of 100 ps, one laser period at 40 MHz. Every regression number refers to it. "Ch1" in the file name is the microscope's detector channel 1, not a name for this decay; other files may carry other channels.
- **Calibration mode / per-decay mode** — the IRF comes from N repetitions of a calibration dye, or from the decay being fitted (section 2).
- **N** — number of calibration repetitions, set by the user.
- **λ** — Whittaker smoothing penalty, in bins⁴. **m** — the cut: last IRF bin and first tail bin. **m₀** — the starting cut from the derivative.
- **Near-best pairs** — (λ, m) pairs within 5 % of the best deviance/dof.
- **Deviance/dof** — Poisson deviance per degree of freedom; about 1 for a fit at the noise level.
- **Block Σz²** — residual-curvature statistic (section 5).

## 1. Role and goal

Build a Python library that fits fluorescence lifetimes from TCSPC decays in batch, called by an existing web GUI.

- **Caller.** A colleague's GitHub repository already holds the GUI scaffolding and its functions. The library must fit that repository's calling conventions (open question, section 10).
- **Division of work.** The library computes; the GUI draws. Return numbers and arrays, never figures. The GUI also owns metadata, exporting and pile-up checks, and passes the library any value it needs, such as the laser repetition rate or the acquisition order.
- **Scope.** Calibration IRF extraction (BIRFI with the λ–m approach), lifetime fitting of samples with that IRF, batch orchestration, and quality flags. Version 2 adds phasor analysis (section 6) and the incomplete-decay model (section 5); version 1 already keeps the data it will need.
- **Basis.** Build from the validated reference implementation (section 8). Respect every finding in section 9; they were measured, not assumed.
- **Excluded.** No monoexponentiality check: the calibration dye is monoexponential by the experimenter's choice.

## 2. Experimental workflow

Every experiment includes a monoexponential calibration dye measured N times, with N set by the user (for example 5 or 10); its IRF serves all samples of that experiment.

1. Measure the calibration dye N times at the start of the experiment, in the same detection channel and filter range as the samples.
2. Check the N repetitions against each other; exclude any that is a different sample (section 4).
3. Sum the accepted repetitions into one high-count decay.
4. Extract the IRF from that sum with BIRFI and the λ–m approach.
5. Save the calibration (IRF, λ, m, reference lifetime, uncertainty, metadata, provenance) as a file.
6. Fit every sample of the experiment by reconvolution with that IRF.
7. Return per-sample results with quality flags. The GUI shows them as a table, and the user inspects only flagged items.

**Why this design.** The IRF no longer comes from the decay it fits. That removes the self-extraction problems found in the per-decay pipeline: τ bias, manufactured residual curvature, and error bars that miss the IRF noise. Samples may be multi-exponential, and each sample needs a single fast fit instead of a full λ–m scan.

**Keep the per-decay mode.** Experiments without a calibration still need the original pipeline: each decay extracts its own IRF.

## 3. Scientific method: BIRFI and the λ–m approach

The IRF is extracted blind from a monoexponential decay, then used to fit lifetimes by Poisson maximum-likelihood reconvolution.

Source: Gómez-Sánchez *et al.*, *Biophysical Reports* **4**, 100155 (2024), including its supporting information and the MATLAB functions `birfi_ls` and `birfi_dec`.

**The identity.** With y = IRF ⊛ x, x monoexponential and the IRF supported on \[0, m\], the tail past m is the underlying decay up to a scalar:

```latex
y(t \ge m) = B\,x(t), \qquad B = \int_0^m e^{s/\tau}\,\mathrm{IRF}(s)\,ds
```

The tail `y[m:]`, re-indexed to start at 0, is therefore x. Knowing y and x, the IRF follows by deconvolution.

**Steps and their validated details.**

- **Background.** The median of the last 10 % of bins (at least 10 bins) of the full histogram, where the fluorescence has died away.
- **Analysis window.** Start at the first bin exceeding 2 % of the peak height above background, minus a margin of 0.8 ns (both configurable; reproduces the 1.0 ns start on the reference decay). The window's first bin is the time origin of the IRF and of the convolution. Record its index in the full-period histogram, so the IRF's absolute position is that index plus the IRF bin, and keep the full-period histogram as well.
- **Rise time.** 10–90 % of the leading edge, measured on the raw counts minus background, by linear interpolation, in bins.
- **Smoothing.** Whittaker, order 2, with the `whittaker-eilers` package: `WhittakerSmoother(lmbda=λ, order=2, data_length=N).smooth(y)`. λ = 0 means no smoothing, since the package needs λ > 0. Never use its `smooth_optimal` (section 9).
- **λ as a cutoff.** The gain is H(ω) = 1/(1 + 16λ sin⁴(ω/2)). Convert a half-gain period T\_c in bins with the exact relation below, valid for T\_c ≥ 2 bins. λ is in bins⁴: rebinning by a factor k needs λ × k⁴.

```latex
\lambda = \left(2\sin(\pi/T_c)\right)^{-4}
```

- **Starting cut m₀.** The minimum of dy/dt after the peak, with `np.gradient` on a trace smoothed at a cutoff of 3 × the rise time. It is only a starting point.
- **Shared boundary bin.** The tail is `y[m:]` and the IRF has m + 1 samples: bin m belongs to both.
- **Deconvolution.** `np.polydiv(y_smoothed, y_smoothed[m:])[0]`, identical to MATLAB `deconv` (forward substitution). It reads only as many tail samples as the IRF's nonzero width.
- **Divisor.** The measured tail, by the user's choice. A fitted exponential is a validated alternative that removes errors-in-variables noise at low counts. The dye's expected lifetime is never used: the analysis stays blind.
- **Normalisation.** Unit area, with negative samples clipped, in both modes. In calibration mode, λ and m are chosen by cross-validation across repetitions (section 4); that selection without clipping was tested and gave a τ₂ bias of −7.0 ± 1.9 ps. Clipping biases τ by about −4 ps only when the IRF returns to zero by the cut (section 9); with a realistic IRF still 3 % of its peak at the cut, no bias was detected. In per-decay mode, the same-data score stays until an honest one is validated, for example splitting each decay's photons at random into two independent halves.
- **λ–m grid.** λ = 0 plus 13 values log-spaced over three decades below the λ whose cutoff is 1.5 × the rise time; m from m₀ − 6 to m₀ + 6. All grid limits are configuration values; the section 8 regression test uses the reference grid, m₀ − 3 to m₀ + 6.
- **Score per pair.** Smooth, cut, deconvolve, normalise, then fit by reconvolution; the score is Poisson deviance per degree of freedom. The lowest score wins, and the lifetime comes from that pair without a refit.
- **Systematic error.** Per-decay mode: half the spread of τ over pairs within 5 % of the best deviance/dof. It is known to be unstable (section 9).

## 4. Calibration stage

The N repetitions are summed into one IRF, and their independence is used to choose λ and m honestly and to measure IRF uncertainty. N is set by the user through the GUI (for example 5 or 10), never fixed, and recorded in the calibration.

- **Combine by summing histograms, then extract once.** Poisson counts add exactly, so the sum has N× the photons and about √N less IRF noise. Do not extract N IRFs and average them: choosing λ and m, and any clipping, are nonlinear and their biases add.
- **Check before summing: is every repetition the same sample?** A mix-up — a wrong file, cuvette or vial — differs from a noisy repetition and must be removed, not just flagged. Compare each repetition's shape with the sum of the others: if the shapes agree, each bin's counts split between the two in a fixed proportion with binomial noise, so Pearson's χ² of that binomial split tests it. Not the binomial deviance: in bins with few counts it over-rejects (1.27 % of clean repetitions at p < 10⁻³ and 2,000 counts peak, against 0.05 % for Pearson, with the same power). No IRF and no lifetime are involved, so the test stays blind and also catches multi-exponential intruders. Remove iteratively: find the repetition with the lowest p; if p < 10⁻³, exclude it and re-test the rest among themselves; repeat until all pass. A single round is not enough, because the wrong sample contaminates the sum each clean repetition is compared with: with a 2.20 ns intruder among five 1.87 ns repetitions, every repetition failed. Identifying the odd one needs N ≥ 3; with N = 2 the test is symmetric, so a mismatch fails the calibration. Separately, check each repetition's timing (centroid) for drift. Validated on simulated sets of five at 10⁴ counts peak: false alarms between 0.05 % and 0.17 % at p < 10⁻³ with Pearson's χ² (0.1 % expected); a sample with a 4 % different lifetime caught (p = 7 × 10⁻²³), one 1.6 % different not. If every repetition is the wrong sample, no comparison among them can tell; only the calibration's own fit quality, an implausible IRF, or the optional post-hoc comparison reveal it.
- **Choose λ and m by cross-validation across repetitions.** Extract the IRF from the sum of N − 1, score it on the held-out repetition, rotate over all N and add the scores. This needs N ≥ 2; with a single calibration decay, fall back to the per-decay scoring and report no IRF error. The IRF is never judged on its own photons, which removes the bias of same-data scoring. *Status: validated and adopted (version 0.3).* On 16 simulated experiments (5 repetitions, smooth IRF), it lowered the IRF error by 0.27 ± 0.05 % compared with choosing on the sum itself (paired; better in 81 %), with no significant lifetime bias. It chooses more smoothing (median λ 0.043 against 0.024), because an IRF scored on photons it never saw gains nothing from fitting their noise.
- **Final IRF.** Extracted from the sum of all accepted repetitions at the chosen (λ, m). The calibration measurement must be scatter-free, e.g. a clear dye solution: scatter in it is folded into the extracted IRF as a derivative-shaped distortion, which every sample then inherits (derived from the deconvolution, not yet measured). BIRFI also assumes a complete decay, so the dye's lifetime must be much shorter than the laser period (reference decay: 1.87 ns at 25 ns, carry-over 1.6 × 10⁻⁶).
- **IRF uncertainty.** Keep the IRFs from leave-one-out subsets (jackknife). Sample fits use them to propagate IRF uncertainty into τ. With N repetitions this error is itself uncertain by roughly 1/√(2(N − 1)): about ±35 % at N = 5 and ±24 % at N = 10. *Status: validated and adopted (version 0.3): across 16 simulated experiments the jackknife matched the scatter the calibration causes (τ₁ 0.0099 vs 0.0096 ns, τ₂ 0.0065 vs 0.0065 ns, a₁ 0.19 vs 0.21 %).*
- **Reference lifetime.** Fit the summed decay blind: the expected lifetime is never an input. Optionally, once the calibration is final, report its difference from the expected value; that difference is never fed back.
- **Convergence diagnostics.** Distance of each pair's IRF from the chosen one, the spread over near-best pairs, and the stability of the peak bin (as in the reference notebook).
- **Saved calibration file.** IRF with its absolute position within the laser period, the summed full-period histogram, λ, m, m₀, rise time, reference lifetime with error, IRF jackknife set, per-repetition checks, acquisition settings as supplied by the GUI (bin width, window, sync, emission filter or wavelength, detector), date, configuration, library version and input-file hashes.

## 5. Sample stage

Each sample is one Poisson maximum-likelihood reconvolution fit with the calibration IRF. The model follows SPCImage NG's decay-parameter panel: 1–3 exponential components plus shift, scatter and offset, each fixable, over a chosen fitting interval.

```latex
\mu(t) = \sum_{i=1}^{n} A_i\,\left(\mathrm{IRF}(t-\delta) \circledast e^{-t/\tau_i}\right)(t) + S\,\mathrm{IRF}(t-\delta) + b, \qquad n = 1, 2, 3
```

Source: SPCImage NG manual, "Model Functions" and its decay-parameter panel (the model is fully specified here; the manual is not required). That panel uses maximum-likelihood estimation, matching the choice validated here.

**Parameters.** Scatter and offset are linear, like the amplitudes, so the fit solves them exactly and searches only the lifetimes and the shift.

| Parameter | Reported as | Role in the fit | Fixable | Default |
| --- | --- | --- | --- | --- |
| Amplitudes a₁–a₃ | % of total, summing to 100 | linear, solved exactly, ≥ 0 | no, as in SPCImage | free |
| Lifetimes τ₁–τ₃ | ns | searched | yes | free |
| Shift δ | bins, sub-bin | searched | yes | free |
| Scatter S | counts; coefficient of the IRF itself | linear, solved exactly, ≥ 0 | yes | fixed at 0 |
| Offset b | counts per bin | linear, solved exactly, ≥ 0 | yes | free |

**Fixing.** A fixed linear parameter has its contribution subtracted and its column removed; a fixed lifetime or shift is left out of the search. Optional starting values are allowed, but the default search needs none.

**Fitting interval \[t₁, t₂\].** Set in ns and snapped to bins; the default is the whole window. Restrict the likelihood, not the model: always convolve over the full window, then sum the deviance and count degrees of freedom only inside the interval. An interval starting at the cut reproduces the tail-only fit, which was unbiased in the simulations; record the interval with every result.

**Interactions to flag.**

- Scatter and a very short lifetime look almost alike, which is likely why SPCImage fixes scatter at 0 by default.
- Offset and a lifetime much longer than the window look alike; lifetimes are capped at the window length.
- An interval that starts after the rise leaves shift and scatter unconstrained: fix them or flag the fit.
- Close lifetimes are strongly correlated (reference decay: τ₂ ± 0.60 ns while the mean is ± 0.0023 ns): flag strong correlations.

**Incomplete decay (version 2).** When the lifetime is not much shorter than the laser period T, photons from earlier pulses add to the decay. Version 1 only raises the `incomplete_decay_suggested` flag; version 2 models it. The model is exact, not a fitting choice: the tails of all earlier pulses form a geometric series, and since the IRF repeats every period, the histogram is the IRF convolved circularly with:

```latex
x_T(t) = \sum_{k \ge 0} e^{-(t+kT)/\tau} = \frac{e^{-t/\tau}}{1 - e^{-T/\tau}}, \qquad 0 \le t < T
```

Verified: identical to summing the tails of 40 earlier pulses, to 3.5 × 10⁻¹⁶. Notes for version 2: report per-pulse amplitudes, dividing out each component's factor 1/(1 − e^(−T/τᵢ)) (1.0000 at 2 ns, 1.0068 at 5 ns, 1.0894 at 10 ns for T = 25 ns), or amplitude fractions are biased. The histogram must span exactly one period, or the model must know where the window sits in it; the model assumes a linear response and a stable period. The default repetition rate is 40 MHz, and the GUI passes another rate. Version 1 already keeps the full-period histograms and the IRF's absolute position.

**Blank measurement (option).** When a blank (buffer only) was measured, add its shape as one more linear term. Its scale is fitted (≥ 0) or fixed from an acquisition-time ratio supplied by the GUI. Never subtract the blank: a difference of counts is no longer Poisson. Without a blank, the flat offset and optional scatter apply as above.

**Fit procedure, errors and reporting.**

- **Discretisation.** Use the same discrete convolution and time origin as the IRF extraction, with no resampling. Sampling the exponential at bins is exact up to a scale absorbed by the amplitude.
- **Time shift δ.** Calibration and samples share channel and filter, so no colour effect is expected. Timing can still drift after the calibration, which is measured only at the start. Fit a sub-bin shift (Fourier phase or interpolation), report it, flag large values, and use the shifts in acquisition order as a drift monitor. Use a Fourier phase shift (tested): with a smooth true IRF shifted exactly and a calibration extracted from simulated repetitions, a 0.3-bin drift was recovered as 0.302 ± 0.004 (cubic spline: 0.303). Its wrap-around matches the physics when the histogram spans one laser period. Linear interpolation is unsuitable: it broadens the IRF as well as shifting it (a 0.4-bin shift of a narrow IRF came out as 0.35, with deviance/dof 1.2).
- **Variable projection, which is maximum likelihood.** For fixed lifetimes and shift, the Poisson likelihood is concave in the linear parameters (amplitudes, scatter, offset), so it has exactly one maximum. Find it by iteratively reweighted least squares: weighted least squares with weights 1/μ from the latest solution, repeated until nothing changes. The fixed point satisfies the Poisson likelihood equations Σᵢ Mᵢₖ (yᵢ − μᵢ)/μᵢ = 0 exactly (to 10⁻¹⁰ on the reference decay), so least squares is only the numerical method. Use `scipy.optimize.nnls` so that A\_i, S, b ≥ 0; the non-negativity is essential, since a negative background made 10 % of low-background fits land in the wrong minimum. Cap the iterations, check that the deviance decreases, and fall back to a direct optimiser if it does not.
- **Lifetime search.** Minimise the profile deviance: a coarse log grid of lifetime combinations (16 values from one bin to the window length), then `scipy.optimize.minimize` (Nelder–Mead in log τ, bounded), with the shift searched alongside unless fixed.
- **Why Poisson ML.** Least squares weighted by the counts biases τ low when bins hold few counts (−95 ps at 100 counts peak and 0.5 counts/bin background). Poisson ML is unbiased in every tested case.
- **Statistical errors (sample photons).** Compute the Fisher information F = Jᵀ diag(1/μ) J at the optimum. J has one row per bin and one column per free parameter, Jᵢⱼ = ∂μᵢ/∂θⱼ: exact for the linear parameters (decay shape, IRF, constant), central differences for lifetimes and shift; diag(1/μ) weights each bin by the inverse of its Poisson variance. The covariance is C = F⁻¹; the errors are √Cⱼⱼ and the correlations Cⱼₖ/(σⱼσₖ), which feed strong\_correlation. Invert the full matrix, because a change in τ is partly compensated by the amplitudes and background. For one exponential this equals the Δdeviance = 1 profile error (1.82 vs 1.80 ps on the reference decay). Propagate to derived quantities with σ\_q² = gᵀCg, g = ∂q/∂θ; the correlations matter (reference decay: τ₂ ± 0.60 ns, yet the mean ± 0.0023 ns). Exclude fixed parameters and parameters at a bound. This error treats the IRF as exact and the model as correct: when deviance/dof is clearly above 1, also report it scaled by √(deviance/dof). It was about 10 % optimistic at 1,000 counts peak.
- **IRF error (calibration photons).** The Fisher error treats the IRF as exact, so the IRF's own noise is added by the jackknife. For each k = 1…N, extract an IRF from the sum of all repetitions except k, at the calibration's λ and m, and refit the sample with it (shift refitted), giving τ₋ₖ. Then σ²\_IRF = ((N − 1)/N) Σₖ (τ₋ₖ − τ̄)², with τ̄ the mean of the τ₋ₖ. The factor is essential: any two leave-one-out IRFs share N − 2 repetitions, so their raw spread is far too small, and this formula restores the variance of the estimate from all N. Sample and calibration photons are independent, so σ\_total = √(σ\_stat² + σ\_IRF²); report all three, for every parameter and derived quantity. λ and m stay fixed (the reference decay's near-best IRFs agree within 2.7 %). Not available at N = 1; σ\_IRF is itself uncertain by about 1/√(2(N − 1)). *Status: validated (section 4): with the combined error, the error scale (rms of error-normalised deviations) was 0.90–0.95 for τ₁, τ₂ and a₁, against 0.98–1.24 for the Fisher error alone.*
- **What to report.** Per component: τ\_i, amplitude and intensity fraction a\_i τ\_i / Σ a\_j τ\_j. Amplitudes also as % of the total (summing to 100, as in SPCImage); shift, scatter and offset with their errors. Also the amplitude- and intensity-weighted mean lifetimes, which stay well determined when the components are not (reference decay: mean 1.8718 ± 0.0023 ns while τ₂ = 2.37 ± 0.60 ns).
- **Residuals.** Pearson residuals, (counts − μ)/√μ, in units of σ.
- **Curvature diagnostic.** Mean residual per 1 ns block × √(bins per block) gives a z-score; report Σz² against the number of blocks, inside the fitting interval. A bin-by-bin runs test missed curvature that this caught.
- **Model choice.** The user chooses 1, 2 or 3 components, as in SPCImage. The library reports what is needed to choose — deviance/dof, block Σz², errors and correlations — and can fit all three side by side. Never report a textbook p-value for adding a component: it is miscalibrated (reference decay: 10⁻⁷ by the textbook, about 0.03 calibrated; section 9). Simulation-calibrated tests and an instrumental echo term are deferred beyond version 1.
- **Validity window.** Warn when the acquisition window is shorter than about 6τ; flag incomplete\_decay\_suggested when earlier pulses add noticeably to the decay; the model that corrects it arrives in version 2.

## 6. Architecture and API

Separate IRF extraction from lifetime fitting, so one calibration serves many samples and the GUI can choose the mode.

**Modules**

- `io` — readers returning a `Decay` (time in ns, counts, and metadata supplied by the GUI), always keeping the full-period histogram rather than only a fitting window. Two-column CSV with a `#` header. Readers sit behind one interface, so other formats can be added later.
- `core` — pure numerical functions with no I/O and no plotting: smoothing, rise time, λ ↔ cutoff, cut detection, deconvolution, reconvolution, deviance, lifetime search, errors, diagnostics.
- `calibration` — `build_calibration(decays, config) -> Calibration`, plus `save` and `load`.
- `fitting` — `fit_sample(decay, calibration, model, config) -> SampleResult`.
- `per_decay` — `analyse(decay, config) -> PerDecayResult`, the current λ–m pipeline with its reference clipping (section 3), for experiments without a calibration.
- `batch` — `run_batch(calibration, decays, config, progress=None, cancel=None) -> BatchResult`. Samples run one after another by default. Parallelism with `concurrent.futures` is optional. A failed sample gets an error status and never stops the batch.
- `config` — one dataclass holding every setting, with defaults equal to the validated values. It is stored with every result. The model is its own object mirroring the GUI panel: number of components, each parameter's value, fixed flag and bounds, and the fitting interval.
- `results` — dataclasses with `to_dict()` returning JSON-safe values. They include the arrays the GUI plots: data, model, residuals, IRF, the λ–m deviance map, the near-best mask and the IRF convergence distances. Internally all times are in ns and every field name carries its unit; display units such as ps are the GUI's choice.

**Package layout.** The library is a multi-file package, one responsibility per file. Besides the modules above it has three small ones: `model` (the model object), `flags` (flag definitions and thresholds) and `simulate` (synthetic decays for tests and experiment planning). The name `tcspc_lifetime` is a placeholder (section 10).

```text
tcspc_lifetime/
├── pyproject.toml
├── README.md
├── src/tcspc_lifetime/
│   ├── __init__.py          public API
│   ├── io/
│   │   ├── __init__.py      reader interface; picks a reader per format
│   │   └── csv_reader.py    two-column CSV
│   ├── core/
│   │   ├── window.py        background, analysis window, rise time
│   │   ├── smoothing.py     Whittaker smoother, λ ↔ cutoff
│   │   ├── cut.py           starting cut m₀ from the derivative
│   │   ├── deconvolution.py BIRFI IRF extraction
│   │   ├── reconvolution.py model, variable projection, deviance, lifetime search
│   │   ├── errors.py        Fisher and jackknife errors
│   │   └── diagnostics.py   block Σz², IRF convergence, correlations
│   ├── model.py             components, fixed flags, bounds, interval, options
│   ├── config.py            every setting with its default
│   ├── calibration.py       build, check, save and load a calibration
│   ├── fitting.py           fit_sample
│   ├── per_decay.py         analyse: λ–m scan on a single decay
│   ├── batch.py             run_batch
│   ├── flags.py             flag codes, thresholds, severities
│   ├── results.py           result dataclasses and to_dict()
│   └── simulate.py          synthetic decays
├── tests/                   one test file per module; data/ holds the reference decay
└── examples/
    └── walkthrough.ipynb    calibration and a small batch, end to end
```

`phasor.py` is reserved for version 2 and is not created in version 1.

**Layering rules.**

- Dependencies point one way. `core` uses only NumPy, SciPy and `whittaker-eilers`; `model`, `config`, `flags` and `results` use nothing beyond the standard library and NumPy; `calibration`, `fitting` and `per_decay` build on `core`; `batch` builds on those; `io` stands alone. No module imports from a layer above it.
- Each file opens with a docstring stating its role in the pipeline and what it depends on.
- The public API is what `__init__.py` exports: `read_decay`, `build_calibration`, `fit_sample`, `analyse`, `run_batch`, and the model, configuration and result classes. Everything else is internal, with a leading underscore.
- Keep files readable in one sitting (about 300 lines as a guideline); split by responsibility when one grows.

**Quality flags.** Each result carries a list of flags (code, severity, message, value, threshold). Initial thresholds are starting points to tune.

| Flag | Trigger (default threshold) | Severity |
| --- | --- | --- |
| `grid_edge` | chosen m is the first or last of its grid, or chosen λ is the largest | warning |
| `curvature` | block Σz² above the 99th percentile of χ² with one degree of freedom per block; to be replaced by a simulation-calibrated threshold | warning |
| `irf_not_converged` | a near-best pair's IRF differs from the chosen one by more than 5 % | warning |
| `low_counts` | peak below 1,000 counts above background | warning |
| `short_window` | window shorter than 6 × the longest lifetime | warning |
| `large_shift` | fitted shift larger than 2 bins | warning |
| `strong_correlation` | any parameter correlation beyond ±0.95 | warning |
| `interval_excludes_rise` | fitting interval starts after the 10 % point of the rise while shift or scatter is free | warning |
| `incomplete_decay_suggested` | carry-over e^(−T/τ) above 1 % (τ > 5.4 ns at 40 MHz); the model that corrects it arrives in version 2 | warning |
| `few_repetitions` | N below 3: with two, a mismatched repetition cannot be identified, and at N = 1 the repetition-based methods are unavailable | warning |
| `repetition_mismatch` | a repetition fails the shape test at p < 10⁻³ after iterative removal; it is excluded from the sum, and the GUI may override | warning |
| `repetition_drift` | a repetition's centroid differs from the others' by more than 0.5 bin | warning |
| `tau_ref_mismatch` | optional, after the calibration is final: lifetime differs from the expected value by more than 3 combined errors | warning |
| `settings_mismatch` | bin width, number of bins or repetition rate differ between calibration and sample | error |
| `fit_failed` | optimiser failure, a non-finite result, or all amplitudes zero | error |
| `calibration_failed` | fewer than 3 repetitions remain after exclusion, or N = 2 with a mismatch | error |

Every threshold is a configuration value; the defaults are starting points to tune on real data.

The drift monitor relies on the acquisition order, which the GUI supplies with each sample.

**Inputs from the GUI.** The calibration decays (their count is N), the sample decays with their acquisition order, the model object (section 5) and the configuration. Optionally: a repetition rate other than 40 MHz, a blank decay with its acquisition-time ratio, and the dye's expected lifetime, used only for the post-hoc comparison.

**Planned module: `phasor`.** IRF-corrected phasor coordinates for each curve and for a whole set of curves; the GUI draws the plots. It ships in version 2. Version 1 already keeps what it needs: each curve's full-period histogram, the background estimate, and the calibration IRF's absolute position within the period. The correction divides each phasor by the IRF's phasor, so it stays blind; at the repetition frequency over one period, incomplete decays need no special treatment.

## 7. Engineering requirements

Readable, reproducible and robust code comes first; with batches of about 100 curves, speed is not a concern.

- **Readability.** Every function has a docstring stating why it is needed, its inputs with units (ns, bins, counts), its output, and where that output is used. Use descriptive names, comment every non-obvious variable, and keep one function per logical step.
- **Dependencies.** NumPy, SciPy and `whittaker-eilers`; matplotlib only for an optional debug-plotting module. Prefer library routines (`np.polydiv`, `scipy.optimize.nnls`, `minimize`, `minimize_scalar`, `brentq`) over hand-written algorithms.
- **Performance.** A typical batch is about 100 curves, and processing time is not a concern. For scale: a per-decay λ–m analysis takes \~4–5 s with one exponential and \~40 s with two; a sample fit with a calibration IRF takes well under 1 s.
- **Robustness.** Validate inputs: empty files, NaN, negative counts, non-uniform bins, picoseconds versus nanoseconds. Give clear error messages and a status for every item.
- **Reproducibility.** Store the configuration, library version, SHA-256 of each input file and all random seeds with every result. Results are deterministic by default.
- **Packaging.** A `pyproject.toml` package with a src/ layout, installable with pip; the tests in tests/ mirror the modules.
- **Tests (pytest).** SI equivalence (`np.polydiv` = `birfi_dec` = `birfi_ls`); exact noise-free recovery; noise calibration per SI Eq. S7; recovery on synthetic decays (Gaussian and two-peak IRFs); the regression numbers of section 8; the false-positive rate of model choice; and error-bar coverage.
- **GUI contract.** The GUI's interface is not known yet. Keep the API framework-agnostic: plain functions, dataclasses and JSON-safe results, with an optional progress callback. Add a thin adapter once the colleague's repository is available.

**Build order.** Each step is tested before the next begins.

1. The package skeleton (section 6 layout) and the core functions, reproducing the section 8 numbers in per-decay mode.
2. The sample model: shift, scatter, offset, fixing, fitting interval and blank, each tested on simulated decays.
3. Calibration mode, including tests of cross-validation across repetitions and of the jackknife IRF error.
4. Batch processing, quality flags and the JSON results contract.
5. Version 2: the phasor module and the incomplete-decay model.

## 8. Reference implementation and regression numbers

The notebook `birfi_lambda_m.ipynb` is the validated reference; the library must reproduce its numbers before anything changes.

**Reference files**

- `birfi_lambda_m.ipynb` — steps 1–5, `analyse_decay`, `print_summary`, `plot_results` (fit, residuals, λ–m map), IRF convergence (`irf_family`, `plot_irf_convergence`) and the synthetic verification.
- `reconvolution.py` — Poisson ML reconvolution with variable projection.
- `birfi.py` and `test_si_equivalence.py` — transcriptions of the SI's MATLAB code, and their tests.
- `whittaker_explorer.ipynb` — interactive view of what λ does.

**Regression values** (per-decay mode with clipping, as section 3 keeps it, and the reference grid m₀ − 3 to m₀ + 6)

| Quantity | Value |
| --- | --- |
| Reference decay, input | `LifeTimeHistogramCh1.csv`: 250 bins of 100 ps, peak ≈ 49,700 counts, background ≈ 27 counts/bin |
| Reference decay, window and rise | starts at 1.0 ns; rise 4.25 bins |
| Reference decay, cut | m₀ = 21; chosen m = 21 (3.1 ns) |
| Reference decay, smoothing | chosen λ = 0.02218 bins⁴ |
| Reference decay, lifetime | τ = 1.8682 ± 0.0018 (stat) ± 0.0057 (syst) ns |
| Reference decay, fit quality | deviance/dof = 1.125 (266.7 / 237) |
| Reference decay, IRF convergence | 24 near-best pairs, IRFs within 2.7 % of the chosen one |
| Synthetic, one decay | two-peak IRF, true τ = 1.2 ns: τ = 1.2024 ± 0.0013 ± 0.0041 ns; IRF error 2.7 % |
| Synthetic, 20 realisations | bias −1.44 ± 0.64 ps; scatter 2.88 ps; coverage 80 % at 1σ, 85 % at 2σ |

**Acceptance criteria.** A build step is complete when its tests meet these limits.

| Test | Pass when |
| --- | --- |
| SI equivalence: `np.polydiv` vs `birfi_dec` vs `birfi_ls` | largest difference below 10⁻⁸ of the peak |
| Noise-free recovery at the true cut | relative IRF error below 10⁻⁶ |
| Whittaker package vs a direct solve of (I + λD₂ᵀD₂)z = y | largest difference below 10⁻¹⁰ of the peak |
| Noise calibration, SI Eq. S7 | mean realised noise within 5 % of the target over 100 realisations |
| Reference-decay regression, per-decay mode, reference grid | λ and m identical; τ within 10⁻⁴ ns of 1.8682; deviance/dof within 0.001 of 1.125 |
| Fisher error vs Δdeviance error, one exponential | agree within 5 % |
| Synthetic recovery, 20 realisations at 5 × 10⁴ peak counts | τ bias within 3 standard errors of zero; mean IRF error below 5 % |
| Error-bar coverage | measured and reported: fraction within 1 and 2 errors, against 68 % and 95 % |
| Model-choice false-positive rate | measured and reported on simulated single-exponential decays |
| Repetition shape test, simulated sets of five at 10⁴ counts peak | false alarms at p < 10⁻³ between 0.05 % and 0.2 %; a sample with a 4 % different lifetime excluded; with a very different sample, only that repetition excluded |

## 9. Validated findings and known pitfalls

Each item below was measured on simulated decays with a known truth, or on the reference decay; the library must not regress on any of them.

- **Clipping biases τ.** Clipping negative IRF values gives τ −4.2 ps too short and residual curvature (block Σz² 37 vs 24 expected) whenever the IRF returns to zero by the cut. Unclipped: −0.1 ps and flat residuals. On the reference decay nothing is clipped, because its IRF is still 4 % of peak at the cut.
- **Unclipped IRFs need honest scoring.** Without clipping, same-data scoring rewards IRFs that absorb noise: on the reference decay the scan drifted to λ = 0 and the edge of the m grid. Hence cross-validation across repetitions (section 4).
- **A self-extracted IRF costs precision.** τ scatter was 3.5 ps with the IRF re-extracted from each decay, against 1.8 ps with the true IRF. A tail-only fit was unbiased at 2.4 ps. This is why samples use a calibration IRF.
- **Error bars.** The Δdeviance error misses IRF noise (1.3 vs 2.9 ps of real scatter). The systematic term ranges from 0.2 to 5.8 ps between realisations, so coverage is 80 % at 1σ and 85 % at 2σ. A whole-pipeline bootstrap gave 2.0 ps from 25 resamples, not yet shown to be calibrated. In calibration mode (version 0.2), Fisher errors of sample fits cover the truth 70 % of the time with a known IRF (68 % expected), but one extracted calibration IRF biases every sample alike: in the example data τ₁ and τ₂ by +14 ps (3–4 standard errors) and the offset by −14 %, all of which vanished with the true IRF. Per-sample errors cannot see this shared bias.
- **Model choice is miscalibrated by the textbook.** On single-exponential data, adding a component improves the deviance by 5.6–6.1 on average instead of about 2. The reference decay's ΔD = 31.7 gives p ≈ 10⁻⁷ by the textbook but p ≈ 0.027 calibrated by simulation (curvature: p ≈ 0.05).
- **The reference decay's tail structure is in the data.** An IRF-free tail fit shows it (Σz² 39.5 vs 21 ± 6), and its lifetime rises from 1.873 to 1.884 ns as the fit start moves from 3.1 to 5.0 ns. A second lifetime and a 0.16 % echo delayed by 5.8 ns explain it equally well.
- **λ selection.** Same-data deviance scanning, with clipping, stayed within 17 % of the best achievable IRF error from 10³ to 3 × 10⁵ counts. The rise-time rule was up to 2.8× worse, no smoothing up to 2.6×, and the package's cross-validation up to 1.5×.
- **Truncated windows.** Least squares over all rows gives a 36 % IRF error when the window covers only 2.2τ, unless the last m rows are dropped. Long division is exact regardless.
- **`whittaker-eilers` `smooth_optimal`.** With `break_serial_correlation=True` it applies a λ found on every fifth point without the 5⁴ rescaling, leaving the full series effectively unsmoothed (gain 0.999 where 0.696 was intended).
- **Colour effect, shift and drift.** Detector timing depends on emission wavelength (the BIRFI paper's motivation); here calibration and samples share channel and filter, which avoids it. Timing drift can still shift the IRF, and with calibration only at the start it is not measured directly.

## 10. Open questions

These decide parts of the design and should be answered before or during the build.

- [x] Same detection channel and filter range for calibration and samples: **yes**. No colour effect; a time shift is still fitted, for drift.
- [x] When the calibration is measured: **at the start of the experiment**. Drift is not measured directly; per-sample shifts monitor it.
- [x] File format: **two-column CSV** for now. More formats may follow, so readers sit behind one interface.
- [ ] GUI interface: **not known yet**. Keep the API framework-agnostic and add a thin adapter once the colleague's repository is available.
- [x] Batch size: **about 100 curves**; processing time is not a concern. Readability and clarity come before speed.
- [x] Sample models: **1–3 exponentials with shift, scatter and offset, each fixable, over a chosen fitting interval**, modelled on SPCImage NG (section 5). Versatility across many sample types is the goal.
- [x] Calibration dye: **known, with a known expected lifetime**, but the analysis stays blind to it to avoid bias.
- [x] Number of calibration repetitions: **set by the user**, for example 5 or 10; never fixed.
- [x] Metadata, exporting and pile-up checks: **handled by the GUI**, which passes the library the values it needs.
- [x] Incomplete decay: **version 2**. Version 1 only flags when it matters; the model follows in version 2, with 40 MHz by default and another rate passed by the GUI.
- [x] Blank measurement: **optional**, used only when one was measured.
- [x] Phasor analysis, per curve and for the whole set: **version 2, in this library, as its own module**. Version 1 keeps each curve's full-period histogram and the calibration IRF's absolute position, so version 2 needs no redesign.
- [x] Library structure: **a multi-file package**, with the layout and layering rules in section 6.
- [ ] Package name: `tcspc_lifetime` is a placeholder; choose the final name.
- [x] Default error bar: **the combined error**, √(statistical² + jackknife IRF²); each part is also reported separately (version 0.3).
- [x] The two proposed methods, tested on 16 simulated experiments: **λ–m by leaving one repetition out — adopted, with clipping** (0.27 ± 0.05 % lower IRF error; without clipping, a 3.7σ lifetime bias); **jackknife IRF error — adopted** (it matched the scatter the calibration causes).
