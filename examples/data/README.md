# Example data (simulated)

Made by `make_example_data.py`; every true value is known.

| Set | Files | Truth |
|---|---|---|
| Calibration | `calibration/cal_01 … cal_05.csv` | monoexponential, τ = 1.87 ns, 10⁴ counts peak, background 5 counts/bin; **`cal_04` is a different sample (τ = 2.20 ns)** and should be excluded |
| Samples | `samples/sample_01 … sample_12.csv` | τ₁ = 0.8 ns, τ₂ = 3.2 ns, amplitudes 60 % / 40 %, 2 × 10⁴ counts peak, background 8 counts/bin, timing drift 0 → 0.4 bins across the series |

All files: 250 bins of 100 ps, two columns (time in ns, counts), `#` header line.
The IRF is smooth (peak 2.2 ns, FWHM 0.45 ns, small bump at 3.0 ns).
