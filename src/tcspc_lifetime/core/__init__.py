"""
Pure numerical functions: no file input/output, no plotting, no state.

Role in the pipeline: everything the calibration, per-decay and sample fits are built from.
window       background, analysis window, rise time
smoothing    Whittaker smoother, λ <-> cutoff, λ grid
cut          starting cut m0, m grid
deconvolution  BIRFI IRF extraction
reconvolution  model, Poisson maximum-likelihood fit, deviance, lifetime search
errors       profile and Fisher errors
diagnostics  residuals, curvature statistic, IRF distances
Depends on: NumPy, SciPy and whittaker-eilers only.
"""
