# Verifying tcspc\_lifetime — version 0.3.1

Oct 7, 2026 · @F

This guide lets you check version 0.3.1 of `tcspc_lifetime` on your own computer: that it installs, passes its tests, reproduces the reference-decay numbers digit for digit, calibrates and fits correctly, refuses what it should refuse, and behaves sensibly on your data. Sections 1–7 check the per-decay mode (build step 1). Sections 8–12 check calibration mode, the sample fits and their three errors, the notebook and the validation study (build steps 2–3). You need Python 3.9 or newer and `tcspc_lifetime_v0.3.1.zip`. Allow about an hour, plus 15 minutes for the slow tests, which can run in the background.

Every expected output below was produced by following this guide in a fresh environment (Python 3.12, NumPy 2.5, SciPy 1.18).

## 1. Install

Unzip `tcspc_lifetime_v0.3.1.zip` and open a terminal in the `tcspc_lifetime` folder, the one containing `pyproject.toml`. Create and activate a fresh environment, so nothing else on your computer interferes:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**These are terminal commands, not Python.** Type them in a terminal — on Windows a Command Prompt or Anaconda Prompt, on macOS the Terminal app, or in VS Code via *Terminal → New Terminal* — and run only the activate line for your system. Pasted into a notebook cell, they give a `SyntaxError`.

**Working only from a notebook instead?** Skip the environment and install into the notebook's own Python. In a cell:

```python
%cd path/to/tcspc_lifetime
%pip install -e ".[test]" matplotlib
```

Then restart the kernel, so Python sees the newly installed package, and run the tests from a cell (the `%cd` again, because a restart resets the folder):

```python
%cd path/to/tcspc_lifetime
import sys
!{sys.executable} -m pytest
```

Install the package with its test tools, plus matplotlib for the plots (add jupyter as well if you do not run notebooks in VS Code):

```bash
pip install -e ".[test]" matplotlib
```

The `-e` makes the install editable: the code you read in `src/` is exactly the code that runs.

**Expected:** pip finishes without errors, and `pip list` shows `tcspc_lifetime 0.3.1`, NumPy, SciPy and `whittaker-eilers 0.2.0`.

## 2. Run the automated tests

The fast tests take under a minute. `pytest` is a terminal command: typed into a notebook cell, it gives `NameError: name 'pytest' is not defined`. **In a notebook**, run it through the notebook's own Python, from the package folder:

```python
%cd path/to/tcspc_lifetime
import sys
!{sys.executable} -m pytest
```

**In a terminal** (environment activated, in the package folder):

```bash
pytest
```

**Expected last line:** `47 passed, 2 deselected`. The deselected ones are the two slow tests, about 15 minutes. Start them in a second terminal and continue with section 3 while they run. In a notebook the equivalent is `!{sys.executable} -m pytest -m slow -s`, but it blocks that notebook until it finishes. In a terminal:

```bash
pytest -m slow -s
```

**Expected:** `1σ coverage: {'tau_1_ns': '70 %', 'tau_2_ns': '70 %', 'amplitude_1_percent': '70 %'}`, then `bias -1.40 ± 0.65 ps, scatter 2.92 ps, mean IRF error 2.99 %`, then `2 passed`. The first is how often a sample fit's 1σ error covers the truth when the IRF is known (target 68 %); the second is the per-decay recovery of 20 simulated decays. The random seeds are fixed, so the numbers should match exactly; a different NumPy version could change the random draws slightly, and the tests still pass within their limits.

What each test file checks:

| Test file | What it checks | Pass condition |
| --- | --- | --- |
| `test_deconvolution.py` | `np.polydiv` equals the SI's MATLAB `birfi_dec` and `birfi_ls`; exact IRF recovery when the cut is where the IRF ends | < 10⁻⁸ of the peak; < 10⁻⁶ |
| `test_smoothing.py` | `whittaker-eilers` solves (I + λD₂ᵀD₂)z = y; λ = 0 leaves the data unchanged; the λ ↔ cutoff relation is exact | < 10⁻¹⁰ of the peak |
| `test_simulate.py` | simulated noise matches the SI's metric (Eq. S7) | within 5 %, at 0.5 % and 2 % |
| `test_reconvolution.py` | the fitted amplitude and background satisfy the Poisson likelihood equations | zero to 10⁻⁶ relative |
| `test_errors.py` | the Fisher error equals the profile error | within 5 % |
| `test_per_decay.py` | the reference-decay numbers, and valid JSON output | λ, m identical; τ within 10⁻⁴ ns; deviance/dof within 0.001 |
| `test_io.py` | reading files, ps conversion, refusing bad input | a clear error message |
| `test_model.py` | the model object refuses inconsistent settings | a clear error message |
| `test_fitting.py` | sample fits with a known IRF: 1 and 2 components, an exact sub-bin shift, fixed parameters, fitting interval and degrees of freedom, scatter, binning mismatch, the three errors | truth within 3 errors; exact counts |
| `test_calibration.py` | the intruder and a slipped-in sample are excluded; refusals with 2 or too few repetitions; cross-validated choice; leave-one-out IRFs; save and load | exact names and counts |
| `test_synthetic_recovery.py` (slow) | 20 simulated decays with known τ = 1.2 ns and a known IRF | bias within 3 standard errors; IRF error < 5 % |
| `test_fitting.py` (slow) | 1σ coverage of sample-fit errors over 40 simulated samples | between 50 % and 85 % |

## 3. Reproduce the reference numbers yourself

The tests compare against stored numbers; here you see the numbers directly. In Python, started from the package folder:

```python
from tcspc_lifetime import read_decay, analyse, PerDecayConfig

decay = read_decay("tests/data/LifeTimeHistogramCh1.csv")
r = analyse(decay, PerDecayConfig.reference_grid())    # the notebook's grid, m0 - 3 ... m0 + 6

print(f"window starts at {r.time_ns[0]:.2f} ns (bin {r.window_start_bin}); rise {r.rise_bins:.2f} bins; m0 = {r.m0_bin}")
print(f"best pair: lambda = {r.best_lambda_bins4:.9f} bins^4, m = {r.best_m_bin}")
print(f"tau = {r.tau_ns:.7f} ns")
print(f"errors: profile {r.tau_profile_error_ns*1e3:.3f} ps, Fisher {r.tau_fisher_error_ns*1e3:.3f} ps, "
      f"systematic {r.tau_systematic_ns*1e3:.3f} ps")
print(f"deviance/dof = {r.deviance_per_dof:.7f}; near-best pairs = {r.near_best_mask.sum()}")
print(f"curvature: sum z^2 = {r.curvature_sum_z2:.1f} over {r.curvature_blocks} blocks")
```

**Expected output**, after about 6 seconds:

```text
window starts at 1.00 ns (bin 10); rise 4.25 bins; m0 = 21
best pair: lambda = 0.022182625 bins^4, m = 21
tau = 1.8681728 ns
errors: profile 1.828 ps, Fisher 1.823 ps, systematic 5.747 ps
deviance/dof = 1.1253595; near-best pairs = 24
curvature: sum z^2 = 55.9 over 24 blocks
```

**Cross-check with the original notebook.** `birfi_lambda_m.ipynb` prints τ = 1.8682 ± 0.0018 (statistical) ± 0.0057 (systematic) ns, at λ = 0.0222 and m = 21. The package agrees in every printed digit; compared function by function, the IRF and the whole deviance map are identical too.

Keep `r` in your session: sections 4–6 use it.

## 4. Look at the fit

The library never draws; plotting is the GUI's job. For your own check:

```python
import matplotlib.pyplot as plt

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]})
top.semilogy(r.time_ns, r.counts, ".", ms=3, label="counts")
top.semilogy(r.time_ns, r.model_counts, "-", label="model")
top.semilogy(r.time_ns[: r.irf.size], r.irf * r.counts.max() / r.irf.max(), "--", label="IRF (scaled to the peak)")
top.set_ylabel("counts per bin"); top.legend()
bottom.plot(r.time_ns, r.residuals_sigma, ".", ms=3); bottom.axhline(0, color="k", lw=0.5)
bottom.set_ylabel("residual (σ)"); bottom.set_xlabel("time (ns)")
plt.show()
```

**What you should see:**

- The model lying on the counts over more than three decades, from the peak (about 49,700 counts) down to the background (about 27 counts per bin).
- An IRF of 22 samples peaking at 2.20 ns, about 0.5 ns wide at half maximum, and back near zero by 3.1 ns, where the tail starts.
- Residuals between −2.7 and +3.0 σ, with an rms of 1.05.

One known feature: the residuals show slow structure. The curvature statistic is 55.9 over 24 blocks, against about 24 ± 7 for a perfect model. This is in the data, not a fault of the package: an IRF-free fit of the tail shows it too (LibraryMasterPrompt, section 9).

## 5. Check the maximum-likelihood claim and the errors

The amplitude and background are found by iterated weighted least squares. The claim (LibraryMasterPrompt, section 5) is that this is exactly Poisson maximum likelihood: at the solution, the derivatives of the log-likelihood vanish. Check it, and compare the two error estimates:

```python
import numpy as np
from tcspc_lifetime.core.reconvolution import decay_shape

shape = decay_shape(r.time_ns, r.irf, r.tau_ns)
print(f"dl/dA = {np.sum(shape * (r.counts / r.model_counts - 1)):.1e}")   # ∂ℓ/∂A
print(f"dl/db = {np.sum(r.counts / r.model_counts - 1):.1e}")             # ∂ℓ/∂b
print(f"Fisher / profile = {r.tau_fisher_error_ns / r.tau_profile_error_ns:.4f}")
```

**Expected:** about −3 × 10⁻¹⁴ and 9 × 10⁻¹¹, which is zero up to rounding (your last digits may differ), and a ratio of 0.9975.

The two errors are independent calculations. One comes from the curvature of the likelihood (the Fisher information); the other scans τ until the deviance rises by 1. Their agreement checks both.

## 6. Check that bad input is refused

A wrong file should fail at once, with a message that says why:

```python
import numpy as np
from tcspc_lifetime import Decay

for label, time_ns, counts in [
    ("negative", np.arange(50) * 0.1, np.r_[np.ones(49), -1.0]),
    ("ps as ns", np.arange(50) * 100.0, np.ones(50)),
    ("uneven",   np.r_[np.arange(49) * 0.1, 5.5], np.ones(50)),
]:
    try:
        Decay(time_ns, counts)
    except ValueError as err:
        print(f"{label}: {err}")
```

**Expected:**

```text
negative: decay: contains negative counts
ps as ns: decay: bin width 100 ns is implausible; is the time column in picoseconds? Use time_unit='ps'.
uneven: decay: bins are not uniformly spaced
```

For a file whose time column is in ps, `read_decay(path, time_unit="ps")` converts it on reading.

Finally, the result must be valid JSON for the GUI:

```python
import json
text = json.dumps(r.to_dict(), allow_nan=False)    # fails if any NaN slipped through
print(len(text))
```

**Expected:** no error, and about 24,000 characters.

## 7. Try a decay of your own

Per-decay mode extracts the IRF from the decay itself, which is valid only for a monoexponential decay: use a measurement of your calibration dye, not a sample. The file needs the same format as the reference: two comma-separated columns, time in ns and counts, with `#` lines as a header. This uses the default settings, with m from m₀ − 6 to m₀ + 6:

```python
mine = analyse(read_decay("path/to/your_decay.csv"))

print(f"tau = {mine.tau_ns:.4f} ns ± {mine.tau_profile_error_ns*1e3:.1f} ps (stat) ± {mine.tau_systematic_ns*1e3:.1f} ps (syst)")
print(f"deviance/dof = {mine.deviance_per_dof:.3f}; curvature {mine.curvature_sum_z2:.1f} over {mine.curvature_blocks} blocks")
print(f"best m = {mine.best_m_bin} (grid {mine.m_grid_bin[0]}..{mine.m_grid_bin[-1]}); "
      f"best lambda = {mine.best_lambda_bins4:.3g} (largest {mine.lambda_grid_bins4[-1]:.3g})")
print(f"near-best IRFs within {100*np.nanmax(mine.irf_distance_map[mine.near_best_mask]):.1f} % of the chosen one")
```

Run on the reference decay, this gives the same pair as before: m = 21 in a grid of 15..27, λ = 0.022183 against a largest value of 1.25, and near-best IRFs within 2.7 %. On your decay, look for:

- **Deviance/dof close to 1.** The reference decay gives 1.13.
- **A best m not at either end of its grid, and a best λ that is not the largest.** Otherwise the grid is too narrow.
- **Near-best IRFs within a few percent of the chosen one.** Otherwise the IRF is poorly determined.
- **Curvature Σz² comparable to the number of blocks.** Much larger means structure the model does not capture.

These checks become automatic quality flags in build step 4.

## 8. Calibrate from the example repetitions

The package ships simulated example data whose true values are known (`examples/data/README.md`). There are five calibration repetitions of a 1.87 ns dye, of which `cal_04` is deliberately a different sample (2.20 ns), and twelve two-component samples (0.8 and 3.2 ns, 60 : 40, with a timing drift of 0 → 0.4 bins across the series). From the package folder:

```python
from pathlib import Path
from tcspc_lifetime import read_decay, build_calibration

cal_files = sorted(Path("examples/data/calibration").glob("*.csv"))
calibration = build_calibration([read_decay(p) for p in cal_files])

print("accepted:", calibration.accepted)
print("excluded:", [(e["name"], f"{e['log10_p_value']:.0f}") for e in calibration.excluded])
print("shape-test p-values:", {k: f"{v:.2g}" for k, v in calibration.shape_test_p_values.items()})
print(f"lambda and m chosen by: {calibration.selection}; lambda = {calibration.best_lambda_bins4:.4g}, "
      f"m = {calibration.best_m_bin}; leave-one-out IRFs: {len(calibration.jackknife_irfs)}")
print(f"dye lifetime (blind): {calibration.tau_reference_ns:.4f} ± {calibration.tau_reference_error_ns:.4f} ns")
```

**Expected output**, after about 15 seconds:

```text
accepted: ['cal_01.csv', 'cal_02.csv', 'cal_03.csv', 'cal_05.csv']
excluded: [('cal_04.csv', '-875')]
shape-test p-values: {'cal_04.csv': '0', 'cal_01.csv': '0.29', 'cal_02.csv': '0.12', 'cal_03.csv': '0.79', 'cal_05.csv': '0.43'}
lambda and m chosen by: cross_validated; lambda = 0.04578, m = 23; leave-one-out IRFs: 4
dye lifetime (blind): 1.8695 ± 0.0020 ns
```

**What to check:**

- **The different sample is excluded:** `cal_04` has log₁₀ p = −875, so its p-value underflows to 0.
- **The clean repetitions pass**, with p-values spread between 0.12 and 0.79, as genuine repetitions of one sample should be.
- **λ and m are chosen by leaving one repetition out,** and four leave-one-out IRFs are kept, one per accepted repetition.
- **The dye's lifetime lands within 1σ of the true 1.87 ns,** although the analysis never used that value: it stays blind.

Keep `calibration` in your session: sections 9 and 10 use it.

## 9. Fit a sample and read its three errors

A two-component model with a separate range for each lifetime, fitted over 1.0–24.9 ns:

```python
from tcspc_lifetime import Model, Parameter, fit_sample

model = Model(2, lifetimes_ns=[Parameter(lower=0.2, upper=1.8), Parameter(lower=1.8, upper=8.0)],
              fit_start_ns=1.0, fit_end_ns=24.9)
r = fit_sample(read_decay("examples/data/samples/sample_01.csv"), calibration, model)

for name in ("tau_1_ns", "tau_2_ns", "amplitude_1_percent", "shift_bins", "offset_counts_per_bin", "tau_mean_intensity_ns"):
    print(f"{name:22s} {r.value(name):8.4f}   stat {r.error(name):.4f}   IRF {r.irf_error(name):.4f}   total {r.total_error(name):.4f}")
print(f"deviance/dof {r.deviance_per_dof:.3f} with {r.degrees_of_freedom} degrees of freedom; "
      f"curvature {r.curvature_sum_z2:.1f} over {r.curvature_blocks} blocks")
```

**Expected output:**

```text
tau_1_ns                 0.8138   stat 0.0118   IRF 0.0074   total 0.0139
tau_2_ns                 3.1989   stat 0.0144   IRF 0.0048   total 0.0152
amplitude_1_percent     60.0037   stat 0.3554   IRF 0.1292   total 0.3782
shift_bins              -0.0039   stat 0.0121   IRF 0.0037   total 0.0126
offset_counts_per_bin    8.3628   stat 0.7697   IRF 0.3958   total 0.8655
tau_mean_intensity_ns    2.5401   stat 0.0055   IRF 0.0040   total 0.0068
deviance/dof 1.170 with 234 degrees of freedom; curvature 22.9 over 24 blocks
```

**The three errors.** *stat* is the Fisher error from the sample's own photons. *IRF* is the jackknife error: the sample refitted with each leave-one-out IRF, measuring the calibration's photon noise, which every sample of the batch shares. *total* is √(stat² + IRF²).

**What to check:**

- **The combination is right.** For τ₁, √(0.0118² + 0.0074²) = 0.0139.
- **The truth lies within about one total error.** τ₁ = 0.8 ns is 1.0 errors away, τ₂ = 3.2 ns 0.1, a₁ = 60 % 0.0, the offset of 8 counts/bin 0.4, and the intensity-weighted mean of 2.5455 ns 0.8. The shift of this first sample is truly 0.
- **The degrees of freedom add up.** The interval holds 240 bins and 6 parameters are free (τ₁, τ₂, shift, two amplitudes, offset), leaving 234.
- **The residuals are flat.** Σz² = 22.9 over 24 blocks, as expected for a correct model.

## 10. Try to break it

A library is trustworthy when it fails in the right way. Six experiments feed it something wrong or unusual:

```python
from tcspc_lifetime import CalibrationError

cal = {p.stem: read_decay(p) for p in cal_files}
sample = read_decay("examples/data/samples/sample_01.csv")

print("--- A: a sample file slipped into the calibration set ---")
c = build_calibration([cal["cal_01"], cal["cal_02"], cal["cal_03"], cal["cal_05"], sample])
print("excluded:", [(e["name"], f"{e['log10_p_value']:.0f}") for e in c.excluded], "accepted:", c.accepted)

print("--- B: three files, one of them a different sample ---")
try: build_calibration([cal["cal_01"], cal["cal_02"], cal["cal_04"]])
except CalibrationError as err: print("CalibrationError:", err)

print("--- C: two files that disagree ---")
try: build_calibration([cal["cal_01"], cal["cal_04"]])
except CalibrationError as err: print("CalibrationError:", err)

print("--- D: fewer repetitions ---")
for names in (["cal_01", "cal_02", "cal_03"], ["cal_01", "cal_02", "cal_03", "cal_05"]):
    r = fit_sample(sample, build_calibration([cal[n] for n in names]), model)
    print(f"N = {len(names)}: tau_1 IRF error {r.irf_error('tau_1_ns'):.4f} ns, total {r.total_error('tau_1_ns'):.4f} ns")

print("--- E: fix tau_1 at its true value ---")
fixed = Model(2, lifetimes_ns=[Parameter(0.8, fixed=True), Parameter(lower=1.8, upper=8.0)], fit_start_ns=1.0, fit_end_ns=24.9)
r = fit_sample(sample, calibration, fixed)
print(f"tau_1 = {r.value('tau_1_ns')} (errors {r.error('tau_1_ns')}, {r.total_error('tau_1_ns')}); "
      f"tau_2 = {r.value('tau_2_ns'):.4f} ± {r.total_error('tau_2_ns'):.4f}; dof {r.degrees_of_freedom}")

print("--- F: the wrong model, one component for a two-component sample ---")
r = fit_sample(sample, calibration, Model(1, fit_start_ns=1.0, fit_end_ns=24.9))
print(f"tau = {r.value('tau_1_ns'):.4f} ns; deviance/dof {r.deviance_per_dof:.1f}; "
      f"curvature {r.curvature_sum_z2:.0f} over {r.curvature_blocks} blocks")
```

**Expected output**, after about a minute:

```text
--- A: a sample file slipped into the calibration set ---
excluded: [('sample_01.csv', '-4119')] accepted: ['cal_01.csv', 'cal_02.csv', 'cal_03.csv', 'cal_05.csv']
--- B: three files, one of them a different sample ---
CalibrationError: calibration_failed: only 2 repetitions remain after excluding ['cal_04.csv']; at least 3 are needed
--- C: two files that disagree ---
CalibrationError: calibration_failed: the two repetitions differ (log10 p = -488.5); with two it cannot be told which one is a different sample. Measure at least three.
--- D: fewer repetitions ---
N = 3: tau_1 IRF error 0.0080 ns, total 0.0142 ns
N = 4: tau_1 IRF error 0.0074 ns, total 0.0139 ns
--- E: fix tau_1 at its true value ---
tau_1 = 0.8 (errors None, None); tau_2 = 3.1867 ± 0.0104; dof 235
--- F: the wrong model, one component for a two-component sample ---
tau = 2.4203 ns; deviance/dof 50.3; curvature 10096 over 24 blocks
```

**What each shows:**

- **A:** the intruder, not a clean repetition, is excluded. This experiment found a real bug while this guide was being written: gross mismatches tied at p = 0, and three clean repetitions were excluded instead. Version 0.3.1 ranks by log p.
- **B and C:** the library refuses rather than guesses: three repetitions minus one is too few, and with two it cannot tell which is wrong.
- **D:** fewer repetitions give a larger IRF error. The effect is modest here, because both calibrations share three of their repetitions.
- **E:** a fixed parameter has no error, and fixing the correlated τ₁ shrinks τ₂'s total error from 0.0152 to 0.0104. The degrees of freedom rise by one.
- **F:** the wrong model is unmistakable: deviance/dof 50 instead of about 1, and Σz² ten thousand instead of about 24.

## 11. Run the notebook

`examples/walkthrough.ipynb` runs the whole workflow the way the GUI will: calibrate, read the samples, choose the model, fit all twelve samples, plot the fits and the distribution of each free parameter. It is shipped with its outputs, so you can compare before and after running it.

Open it in VS Code or Jupyter with the `examples` folder as the working directory (the default when you open the notebook there), select the environment from section 1 as the kernel, and run all cells. It takes about a minute.

**Expected printed lines** (among others):

```text
EXCLUDED cal_04.csv: shape test log10 p = -875 -> a different sample
λ and m chosen by: cross_validated; leave-one-out IRFs kept: 4
calibration dye lifetime (blind): 1.8695 ± 0.0020 ns
trend: +0.394 bins from first to last sample
```

**What the figures should show:**

- **Calibration:** `cal_04` dashed in red and visibly slower; the IRF with its four leave-one-out IRFs in grey, differing only in the small structure before the cut; a score map with a valley at cuts 22–25 and the star at λ ≈ 0.046, m = 23.
- **Fits:** the model on the counts over more than three decades, and residuals within about ±3σ with no structure, for all twelve samples.
- **Distributions:** each mean close to the red dashed true value. The spread (SD) should be close to the mean statistical error, since the samples differ only by photon noise, except for the shift, whose spread is the simulated drift.
- **Drift monitor:** the shift rising from about 0 to about 0.4 bins across the series, matching the simulated drift.

Then change the model in section 3 of the notebook, for example to `n_components=1`, and run sections 4–5 again. The fits should fail visibly, with curved residuals and deviance/dof around 50.

## 12. Review the validation study

Two choices in version 0.3 rest on a simulation study: choosing λ and m by leaving one repetition out (with clipping), and the jackknife IRF error. `validation/README.md` summarises it. The study simulated 16 independent experiments, each with 5 calibration repetitions and 4 two-component samples, and compared three ways of choosing λ and m against the known truth.

To see the full numbers without re-running anything (a few seconds):

```bash
cd validation
python analyse_calibration_methods.py
```

**Expected**, among other lines:

```text
16 realisations x 4 samples

== same_clip: IRF error 3.41 ± 0.11 %   calibration tau bias -1.9 ± 0.9 ps   grid edge 0 %
```

…followed by `cv_free` at 3.74 ± 0.18 % and `cv_clip` at 3.14 ± 0.13 %, each with one line per parameter.

**What to look for in each method's block:**

- **IRF error:** `cv_clip` (the adopted method) is lowest. Paired over the same experiments it is 0.27 ± 0.05 % better than `same_clip`, the version 0.2 method.
- **Bias:** `cv_free`, the original no-clipping proposal, has a τ₂ bias of −0.0070 ± 0.0019 ns. This is why it was rejected.
- **Jackknife:** the last two numbers on each parameter line should agree — the scatter the calibration actually causes against the mean jackknife IRF error. For `cv_clip`: τ₁ 0.0096 vs 0.0099, τ₂ 0.0065 vs 0.0065.
- **Error scale:** under "rms z", the Fisher error alone, then combined with the jackknife. Combined, it should sit close to 1.

To reproduce the study from scratch, run `python calibration_methods.py 0 16 1` in the same folder, then the analysis again. It takes about 25 minutes on one core. The seeds are fixed, so it rewrites the results file with the same numbers.

## 13. Read the code

The code is meant to be read: every function's docstring states why it exists, its inputs with units, its output, and where that output is used. A sensible order:

1. `src/tcspc_lifetime/per_decay.py`: `analyse` is the whole per-decay pipeline in about 60 lines, each calling one core function.
2. `core/window.py`, `core/smoothing.py`, `core/cut.py`: background, window, rise time, λ, the starting cut and the grids. Compare with the LibraryMasterPrompt, section 3.
3. `core/deconvolution.py`: BIRFI itself, one call to `np.polydiv`, plus the normalisation.
4. `core/reconvolution.py`: the model, the iterated maximum-likelihood fit with its safeguard, the lifetime search, and the IRF placement and Fourier shift used by sample fits.
5. `core/errors.py` and `core/diagnostics.py`, including the repetition shape test (Pearson's χ², ranked by log p).
6. `model.py`: the settings object behind the GUI's model panel. Compare with the parameter table in section 5 of the prompt.
7. `calibration.py`: `build_calibration` — the binning check, excluding mismatched repetitions, the cross-validated choice of λ and m, and the leave-one-out IRFs. Compare with section 4.
8. `fitting.py`: `fit_sample` and its helper class (from the search vector to the model, the exact linear solve, the coarse search, the refinement, the errors), then `_add_irf_errors`. Compare with section 5.
9. `batch.py`, `results.py` (`SampleResult`, `FailedFit`) and `io/`.

Allow about an hour for the whole tour.

Two structural checks from the prompt's layering rules — the core imports nothing from the rest of the package, and nothing draws:

```bash
grep -rn "^from\|^import" src/tcspc_lifetime/core
grep -rn "matplotlib" src/
```

**Expected:** the first shows only `numpy`, `scipy`, `whittaker_eilers` and imports between core files; the second shows nothing.

## 14. Checklist

- [ ] Installed in a fresh environment without errors (section 1)
- [ ] `pytest`: 47 passed, 2 deselected (section 2)
- [ ] `pytest -m slow -s`: 2 passed, with coverage and recovery as in section 2
- [ ] Reference numbers reproduced (section 3)
- [ ] Fit, IRF and residuals look as described (section 4)
- [ ] Likelihood derivatives about zero; Fisher / profile about 1 (section 5)
- [ ] Bad input refused; the result serialises to JSON (section 6)
- [ ] A decay of your own analysed in per-decay mode (section 7)
- [ ] Calibration excludes `cal_04` and stays blind (section 8)
- [ ] Sample fit within about one total error of the truth (section 9)
- [ ] All six break-it experiments behave as described (section 10)
- [ ] Notebook runs; figures as described (section 11)
- [ ] Validation numbers reviewed (section 12)
- [ ] Code read; both layering checks pass (section 13)

**If something fails,** send back three things: the command or snippet you ran, its complete output, and your versions:

```bash
python -V
pip list | grep -i -E "numpy|scipy|whittaker|tcspc"          # macOS / Linux
pip list | findstr /i "numpy scipy whittaker tcspc"            # Windows
```
