# Validation: choosing λ and m, and the IRF error, in calibration mode

`calibration_methods.py` simulates 16 independent experiments, each with 5 calibration
repetitions (τ = 1.87 ns, 10⁴ counts peak, background 5) and 4 two-component samples
(0.8 / 3.2 ns, 60 : 40, 2 × 10⁴ peak, background 8), with a smooth known IRF. It compares
three ways of choosing λ and m for the calibration IRF, and checks the jackknife IRF error.
Results: `calibration_methods_results.jsonl`; summary: `analyse_calibration_methods.py`.
Run: `python calibration_methods.py 0 16 1` (about 25 minutes on one core).

| Method | IRF error | Outcome |
|---|---|---|
| λ–m chosen on the summed decay, clipped (version 0.2) | 3.41 ± 0.11 % | baseline |
| λ–m by leaving one repetition out, **clipped** | 3.14 ± 0.13 % | **adopted**: 0.27 ± 0.05 % better (paired, 81 % of calibrations), no significant bias |
| λ–m by leaving one repetition out, not clipped | 3.74 ± 0.18 % | rejected: τ₂ bias −7.0 ± 1.9 ps |

Jackknife IRF error (adopted method): it matches the scatter that the calibration causes
(τ₁ 0.0099 vs 0.0096 ns, τ₂ 0.0065 vs 0.0065 ns, a₁ 0.19 vs 0.21 %). Combined with the Fisher
error, the error scale (rms of normalised deviations) is 0.95 / 0.90 / 0.93 for τ₁ / τ₂ / a₁,
against 1.24 / 0.98 / 1.09 for the Fisher error alone.
