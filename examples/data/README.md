# Example data

Two sets: simulated decays with known truth (Part I of the walkthrough) and measured dyes (Part II).

## Simulated

Made by `make_example_data.py`; every true value is known.

| Set | Files | Truth |
|---|---|---|
| Calibration | `calibration/cal_01 … cal_05.csv` | monoexponential, τ = 1.87 ns, 10⁴ counts peak, background 5 counts/bin; **`cal_04` is a different sample (τ = 2.20 ns)** and should be excluded |
| Samples | `samples/sample_01 … sample_12.csv` | τ₁ = 0.8 ns, τ₂ = 3.2 ns, amplitudes 60 % / 40 %, 2 × 10⁴ counts peak, background 8 counts/bin, timing drift 0 → 0.4 bins across the series |

All files: 250 bins of 100 ps, two columns (time in ns, counts), `#` header line.
The IRF is smooth (peak 2.2 ns, FWHM 0.45 ns, small bump at 3.0 ns).

## Measured: `measured/`

Three dyes in solution, measured in the same detection channel with the same TCSPC settings.
Files are unchanged exports of the acquisition software, regrouped into folders without spaces.

| Folder | Dye | Files | Peak counts | Role |
|---|---|---|---|---|
| `calibration_atto655/` | ATTO655, series 1 ("for IRF") | 6 | ≈ 6 × 10⁴ | calibration |
| `atto655/` | ATTO655, series 2 ("for lifetime") | 6 | ≈ 6 × 10⁴ | independent check of the calibration |
| `af647_phalloidin/` | Alexa Fluor 647–phalloidin, 100× | 5 | ≈ 7 × 10⁵ | sample |
| `star_red/` | Abberior STAR RED, series 2, 100× | 5 | ≈ 2 × 10⁵ | sample |

All files: 250 bins of 100 ps (25 ns window), two columns (time in ns, counts), `#` header line.
No ground truth; what the library finds is in the walkthrough, Part II.

Acquisition details (to be filled in):

- Instrument and detector: Abberior Stedycon Ch1
- Excitation (wavelength, repetition rate): 640nm 10uW
- Emission filter: 650-700nm
- Solvent, concentration, temperature: mqH2O, 21C
