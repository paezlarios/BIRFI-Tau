"""
tcspc_lifetime - fluorescence lifetimes from TCSPC decays, built to be called by a GUI.

Version 1, build steps 1-3:
    per-decay mode   the IRF is extracted blindly from the decay being fitted (BIRFI, λ-m)
    calibration mode the IRF comes from N repetitions of a calibration dye (mismatched
                     repetitions excluded; λ and m chosen by leaving one repetition out);
                     samples are fitted with 1-3 exponentials, shift, scatter and offset,
                     each fixable, with ranges and a fitting interval; errors combine the
                     sample's photon noise and the calibration IRF's (jackknife)
Not yet: quality flags (step 4).

Public API
    read_decay(path, time_unit="ns", name=None, metadata=None) -> Decay
    analyse(decay, config=None) -> PerDecayResult                      per-decay mode
    build_calibration(decays, config=None) -> Calibration              calibration mode
    fit_sample(decay, calibration, model=None) -> SampleResult
    run_batch(decays, calibration, model, progress=None) -> list of SampleResult / FailedFit
    Model, Parameter, Decay, Calibration, CalibrationError, PerDecayConfig
"""
__version__ = "0.4.0"

from .config import PerDecayConfig
from .io import Decay, read_decay
from .per_decay import analyse
from .results import FailedFit, PerDecayResult, SampleResult
from .model import Model, Parameter
from .calibration import Calibration, CalibrationError, build_calibration
from .fitting import fit_sample
from .batch import run_batch

__all__ = ["read_decay", "analyse", "build_calibration", "fit_sample", "run_batch", "Model",
           "Parameter", "Decay", "Calibration", "CalibrationError", "PerDecayConfig",
           "PerDecayResult", "SampleResult", "FailedFit", "__version__"]
