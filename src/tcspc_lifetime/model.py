"""
The sample model: what is fitted, what is fixed, within which ranges, over which interval.

Role in the pipeline: mirrors the GUI's decay-parameter panel, modelled on SPCImage NG:
1-3 exponential components plus shift, scatter and offset, each fixable, fitted over a
chosen interval. Amplitudes are always free (as in SPCImage). Ranges apply to lifetimes and
shift; amplitudes, scatter and offset are only kept >= 0.
Depends on: the standard library only.
"""
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Parameter:
    """
    One adjustable parameter of the model.

    value         starting value of the fit, or the value used when fixed
                  (None: found automatically by the coarse search)
    fixed         True keeps the parameter at `value`
    lower, upper  allowed range (None: the parameter's default range)
    """
    value: Optional[float] = None
    fixed: bool = False
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self):
        if self.fixed and self.value is None:
            raise ValueError("a fixed parameter needs a value")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(f"lower bound {self.lower} must be below upper bound {self.upper}")
        if self.value is not None and not self.fixed:
            if ((self.lower is not None and self.value < self.lower) or
                    (self.upper is not None and self.value > self.upper)):
                raise ValueError(f"starting value {self.value} lies outside [{self.lower}, {self.upper}]")


@dataclass
class Model:
    """
    The model fitted to every sample of a batch.

    n_components  1, 2 or 3 exponential components
    lifetimes_ns  one Parameter per component (ns). Default: free, range one bin to the
                  length of the fitting interval. Give each component its own range to keep
                  the component labels stable across a batch.
    shift_bins    IRF shift relative to the calibration (bins). Default: free in [-3, 3].
    scatter       coefficient of the IRF itself, i.e. an instantaneous component (counts).
                  Default: fixed at 0, as in SPCImage.
    offset        constant background (counts per bin). Default: free.
    fit_start_ns, fit_end_ns
                  fitting interval (ns). Default: from the calibration window's start to the
                  end of the histogram. Only bins inside it enter the likelihood; the model
                  is always computed over the whole histogram.
    """
    n_components: int = 1
    lifetimes_ns: List[Parameter] = field(default_factory=list)
    shift_bins: Parameter = field(default_factory=lambda: Parameter(0.0, False, -3.0, 3.0))
    scatter: Parameter = field(default_factory=lambda: Parameter(0.0, True))
    offset: Parameter = field(default_factory=Parameter)
    fit_start_ns: Optional[float] = None
    fit_end_ns: Optional[float] = None

    def __post_init__(self):
        if self.n_components not in (1, 2, 3):
            raise ValueError("n_components must be 1, 2 or 3")
        if not self.lifetimes_ns:
            self.lifetimes_ns = [Parameter() for _ in range(self.n_components)]
        if len(self.lifetimes_ns) != self.n_components:
            raise ValueError(f"{self.n_components} components need {self.n_components} lifetime "
                             f"parameters, got {len(self.lifetimes_ns)}")
        for k, tau in enumerate(self.lifetimes_ns, start=1):
            if (tau.value is not None and tau.value <= 0) or (tau.lower is not None and tau.lower <= 0):
                raise ValueError(f"lifetime {k}: values and lower bound must be positive")
        for name in ("scatter", "offset"):
            parameter = getattr(self, name)
            if parameter.upper is not None or (parameter.lower not in (None, 0, 0.0)):
                raise ValueError(f"{name}: ranges are not supported; it is only kept >= 0")
            if parameter.value is not None and parameter.value < 0:
                raise ValueError(f"{name} cannot be negative")
        if (self.fit_start_ns is not None and self.fit_end_ns is not None
                and self.fit_start_ns >= self.fit_end_ns):
            raise ValueError("fit_start_ns must be before fit_end_ns")

    def to_dict(self):
        """All settings as a plain dictionary (stored with every result)."""
        return asdict(self)
