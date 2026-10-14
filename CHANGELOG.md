# Changelog

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
