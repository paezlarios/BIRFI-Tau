# Changelog

## Unreleased
- Measured test data in `examples/data/measured/`: ATTO655 (calibration, 6 repetitions, and
  an independent series), Alexa Fluor 647–phalloidin and Abberior STAR RED. The walkthrough
  has a Part II that analyses them; its saved outputs are regenerated (Part I's were from
  0.3.1 with `fit_end_ns=24.9`). New regression test `tests/test_measured_dyes.py`: the second
  ATTO655 series refits to within 1 % of the calibration's lifetime. No library change.

## 0.4.0
- The IRF on the full histogram, `irf_full_period`: the unit-area IRF at its absolute
  position, zeros elsewhere, one value per bin. In `PerDecayResult`; in `SampleResult` as
  the fit used it, shifted by `shift_bins` (sample results carried no IRF before); and as a
  `Calibration` property, included in `to_dict()` but not written by `save()` (0.3.1 refuses
  unknown keys) and ignored by `load()`, so calibration files stay readable by both
  versions. The compact `irf` is unchanged and no fitted number changes.
- Documentation only, no change in behaviour. `docs/LibraryMasterPrompt.md` and
  `docs/VERIFICATION.md`, listed in `docs/README.md` but missing, are now exported from Claude
  Docs. The clipping notes in `normalise_irf` and `PerDecayConfig` now say what the code does:
  negative IRF samples are clipped in both modes. `reference/README.md` says where the
  reference decay is; the README's status line names version 0.3.1.

## 0.3.1
- Repetition shape test returns log10 p, computed through the Wilson–Hilferty approximation,
  and repetitions are ranked by it. Before, gross mismatches tied at p = 0 and clean
  repetitions could be excluded instead of the intruder (found while writing the
  verification guide). Regression test added.

## 0.3.0
- IRF calibration chooses λ and m by leaving one repetition out (with clipping): 0.27 ± 0.05 %
  lower IRF error than choosing on the summed decay, on 16 simulated experiments.
  The no-clipping variant was rejected (τ₂ bias −7.0 ± 1.9 ps).
- Jackknife IRF error from the leave-one-out IRFs; every sample parameter reports a
  statistical, an IRF and a combined error.
- Validation study in `validation/`.

## 0.2.0
- Calibration mode: N repetitions, blind exclusion of mismatched ones (Pearson χ² shape test),
  IRF from their sum.
- Sample fits: 1–3 exponentials, shift (Fourier), scatter, offset, each fixable, with ranges
  and a fitting interval; Fisher errors; derived fractions and mean lifetimes.
- `run_batch`, example data with known true values, `examples/walkthrough.ipynb`.

## 0.1.0
- Per-decay mode: BIRFI with the λ–m approach, reproducing the reference notebook bit for bit.
- Reading CSV decays with validation; JSON-safe results; tests against the BIRFI SI.
