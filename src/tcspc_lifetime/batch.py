"""
run_batch: one calibration and one model, many samples.

Role in the pipeline: what the GUI calls for a batch. Samples are fitted one after another;
a sample that fails is recorded as a FailedFit and never stops the batch. Quality flags are
added in build step 4.
Depends on: fitting, results.
"""
from .fitting import fit_sample
from .results import FailedFit


def run_batch(decays, calibration, model, progress=None):
    """
    Fit every sample with the same calibration and model.

    Input:  decays      - list of io.Decay (the samples, in acquisition order)
            calibration - calibration.Calibration
            model       - model.Model
            progress    - optional function called as progress(done, total, name)
    Output: list with one SampleResult or FailedFit per sample, in the same order
    Used by: the GUI.
    """
    results = []
    for k, decay in enumerate(decays):
        try:
            results.append(fit_sample(decay, calibration, model))
        except Exception as error:                      # one bad sample must not stop the batch
            results.append(FailedFit(decay_name=decay.name, error_message=str(error)))
        if progress is not None:
            progress(k + 1, len(decays), decay.name)
    return results
