# tcspc_lifetime

Fluorescence lifetimes from TCSPC decays, with the instrument response function (IRF)
extracted blindly from a monoexponential decay (BIRFI, Gómez-Sánchez *et al.*, Biophys. Rep.
4, 100155, 2024) and the λ–m approach. Built to be called by a GUI for batch processing.
The full specification is the LibraryMasterPrompt (`docs/LibraryMasterPrompt.md`).

## Status: version 0.4.0 — build steps 1–3

- **Per-decay mode** (step 1): the IRF comes from the decay being fitted.
- **Calibration mode** (steps 2–3): the IRF comes from N repetitions of a calibration dye.
  Mismatched repetitions (a different sample) are detected blindly and excluded; λ and m
  are chosen by leaving one repetition out; the IRF is extracted from the sum. Samples are
  fitted with 1–3 exponentials plus shift, scatter and offset, each fixable, with ranges
  and a fitting interval, by Poisson maximum likelihood. Every parameter gets a
  statistical error, a jackknife IRF error and their combination.
- **Validation** of the calibration methods: `validation/README.md`.
- **Not yet:** quality flags (step 4).

## Install and test

    pip install -e ".[test]"
    pytest              # fast tests (under a minute)
    pytest -m slow -s   # error-bar coverage and 20-decay recovery (about 10 minutes)

## Use

    from tcspc_lifetime import read_decay, build_calibration, Model, Parameter, run_batch

    calibration = build_calibration([read_decay(p) for p in calibration_paths])
    model = Model(2, lifetimes_ns=[Parameter(lower=0.2, upper=1.8), Parameter(lower=1.8, upper=8)])
    results = run_batch([read_decay(p) for p in sample_paths], calibration, model)
    results[0].value("tau_1_ns"), results[0].error("tau_1_ns")
    payload = results[0].to_dict()        # JSON-safe, for the GUI

`examples/walkthrough.ipynb` runs the whole workflow on simulated example data with known
true values (`examples/data/`, made by `examples/make_example_data.py`).

All times are in ns; every result field name carries its unit.

## Repository layout

| Folder | Contents |
|---|---|
| `src/tcspc_lifetime/` | the library (the only code that is installed) |
| `tests/` | fast tests; `pytest -m slow` adds the long simulations |
| `examples/` | the walkthrough notebook and its simulated data |
| `validation/` | the simulation study behind the calibration methods |
| `docs/` | the specification (LibraryMasterPrompt), the verification guide, and `FILES.md`, a guide to every file |
| `reference/` | the BIRFI work the library was built from; source of the regression numbers |

Changes are listed in `CHANGELOG.md`. Each released version is tagged (`v0.3.1`, ...);
applications that embed the library, such as TACO, copy a tagged version unmodified.
