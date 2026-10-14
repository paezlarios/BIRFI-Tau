# File guide

Every file in the repository and what it is for. Inside `src/`, each module also opens with
a docstring stating its role in the pipeline and what it depends on, and every function's
docstring states why it exists, its inputs with units, its output and where it is used.

## Top level

| File | Purpose |
| --- | --- |
| `README.md` | what the library does, its status, how to install, test and use it, the repository layout |
| `CHANGELOG.md` | what changed in each version |
| `pyproject.toml` | package definition: name, version, dependencies, test settings |
| `.gitignore` | files Git leaves out (caches, environments, build output) |
| `.github/workflows/tests.yml` | runs the fast tests on every push, on Linux and Windows |

## The library: `src/tcspc_lifetime/`

Read in this order; the first column is also the order data flows through the pipeline.

| File | Purpose | Main contents |
| --- | --- | --- |
| `__init__.py` | the public API and the version | `read_decay`, `analyse`, `build_calibration`, `fit_sample`, `run_batch`, `Model`, `Parameter`, `Calibration`, results |
| `io/__init__.py` | reading decays | `Decay` (validated histogram), `read_decay` (picks a reader by file suffix), `READERS` |
| `io/csv_reader.py` | the two-column CSV format | `read_csv_columns`, with ns/ps conversion |
| `config.py` | every setting of the λ–m pipeline, with defaults | `PerDecayConfig`, `reference_grid()` |
| `core/__init__.py` | overview of the numerical core | — |
| `core/window.py` | background, analysis window, rise time | `estimate_background`, `find_window_start`, `measure_rise_time` |
| `core/smoothing.py` | Whittaker smoothing and λ | `whittaker_smooth`, `lambda_for_cutoff`, `cutoff_for_lambda`, `make_lambda_grid` |
| `core/cut.py` | where the IRF ends | `find_initial_cut` (m₀), `make_m_grid` |
| `core/deconvolution.py` | BIRFI itself | `deconvolve_irf` (polynomial long division), `normalise_irf` |
| `core/reconvolution.py` | the model and its maximum-likelihood fit | `decay_shape`, `poisson_deviance`, `fit_linear_parameters` (Poisson ML for linear parameters), `fit_single_lifetime`, `place_irf` (IRF position and Fourier shift), `exponential_component` |
| `core/errors.py` | statistical errors | `profile_error` (Δdeviance = 1), `fisher_covariance`, `central_difference` |
| `core/diagnostics.py` | checks on a fit or a calibration | `pearson_residuals`, `block_curvature`, `irf_distance`, `repetition_shape_test` |
| `per_decay.py` | per-decay mode: the IRF from the decay itself | `analyse` (the λ–m scan) |
| `model.py` | the sample model's settings | `Model`, `Parameter` |
| `calibration.py` | the IRF from N repetitions of a calibration dye | `build_calibration`, `Calibration` (with `save`/`load`), `CalibrationError`, `irf_at` |
| `fitting.py` | fitting one sample with a calibration IRF | `fit_sample` (with statistical, IRF and combined errors) |
| `batch.py` | a simple loop over samples (TACO runs its own loop instead) | `run_batch` |
| `results.py` | result objects, convertible to JSON | `PerDecayResult`, `SampleResult`, `FailedFit`, `json_safe` |
| `simulate.py` | synthetic decays with known truth | `smooth_irf`, `two_peak_irf`, `expected_decay`, `expected_sample`, `poisson_decay`, `noise_percent` |
| `flags.py` | placeholder for quality flags (next build step) | — |

## Tests: `tests/`

| File | What it checks |
| --- | --- |
| `conftest.py` | shared fixtures: the reference decay and its analysis |
| `data/LifeTimeHistogramCh1.csv` | the reference decay (detector channel 1 of the microscope) |
| `test_io.py` | reading files, ps conversion, refusing bad input |
| `test_smoothing.py` | the smoothing package against a direct solve; λ = 0; the λ ↔ cutoff relation |
| `test_deconvolution.py` | equivalence with the BIRFI paper's MATLAB code; exact IRF recovery |
| `test_simulate.py` | the simulator's noise matches the paper's metric |
| `test_reconvolution.py` | fitted parameters satisfy the Poisson likelihood equations |
| `test_errors.py` | Fisher and profile errors agree |
| `test_per_decay.py` | the reference-decay regression numbers; JSON output |
| `test_model.py` | the model refuses inconsistent settings |
| `test_fitting.py` | sample fits recover known values; shift; fixed parameters; interval; scatter; the three errors; error coverage (slow) |
| `test_calibration.py` | excluding a different sample; refusals; cross-validated λ and m; leave-one-out IRFs; save and load |
| `test_synthetic_recovery.py` | recovery over 20 simulated decays (slow) |

## Examples: `examples/`

| File | Purpose |
| --- | --- |
| `walkthrough.ipynb` | the whole workflow on example data, saved with its outputs |
| `make_example_data.py` | writes the example data |
| `data/README.md` | the example data's true values |
| `data/calibration/cal_01 … cal_05.csv` | five calibration repetitions; `cal_04` is deliberately a different sample |
| `data/samples/sample_01 … sample_12.csv` | twelve two-component samples with a timing drift |

## Validation: `validation/`

| File | Purpose |
| --- | --- |
| `README.md` | the study's question, results and how to rerun it |
| `calibration_methods.py` | the study: 16 simulated experiments, three ways of choosing λ and m, and the jackknife |
| `calibration_methods_results.jsonl` | its results, one line per experiment |
| `analyse_calibration_methods.py` | summarises those results |

## Reference: `reference/`

The BIRFI work the library was built from; `reference/README.md` explains each file.

## Documentation: `docs/`

| File | Purpose |
| --- | --- |
| `README.md` | index of the documentation |
| `FILES.md` | this guide |
| `LibraryMasterPrompt.md` | the specification (exported from Claude Docs) |
| `VERIFICATION.md` | how to test and review a version (exported from Claude Docs) |
