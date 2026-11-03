# Reference implementation (before the library)

The work the library was built from. It is kept because the library's regression numbers
come from it: `birfi_lambda_m.ipynb` is the validated per-decay pipeline whose results the
library reproduces bit for bit (`tests/test_per_decay.py`).

| File | What it is |
|---|---|
| `birfi_lambda_m.ipynb` | the λ–m pipeline, step by step, on the reference decay; source of the regression numbers |
| `reconvolution.py` | the original Poisson maximum-likelihood reconvolution fit |
| `birfi.py`, `test_si_equivalence.py` | transcriptions of the BIRFI paper's MATLAB code (Gómez-Sánchez *et al.*, Biophys. Rep. 4, 100155, 2024) and their checks |
| `birfi_demo.py`, `fig*.png` | reproductions of the paper's figures |
| `analyse_ch1.py`, `ch1_analysis.png`, `irf_convergence_ch1.png` | the first analyses of the reference decay |
| `whittaker_explorer.ipynb`, `birfi_numpy.ipynb` | interactive explorations of the smoothing and of the method |
| `../tests/data/LifeTimeHistogramCh1.csv` | the reference decay (detector channel 1 of the microscope); kept only in `tests/data/`. The notebooks read it from their own folder and `analyse_ch1.py` from a fixed path, so copy it next to them, or change the path, to rerun them |
| `README_birfi.md` | the original documentation of these files |

These files are not part of the installed package and are not maintained; the library in
`src/` is the maintained code.
