"""
Every setting of the analysis, with its default.

Role in the pipeline: one object holds all the knobs, so a result can always say exactly
how it was produced (it is stored with every result). Defaults are the validated values of
the reference implementation, except the m grid, which by the user's choice spans m0 - 6 to
m0 + 6; `PerDecayConfig.reference_grid()` restores m0 - 3 to m0 + 6 for the regression test.
Depends on: the standard library only.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PerDecayConfig:
    """
    Settings of the per-decay mode (the IRF is extracted from the decay being fitted).

    background_tail_fraction  fraction of the last bins whose median is the background
    window_margin_ns          the analysis window starts this long before the rise (ns)
    rise_threshold            the rise is the first bin above this fraction of the peak
    cut_smoothing_factor      the starting cut uses smoothing with a cutoff of this x rise time
    lambda_top_factor         the largest λ has a cutoff of this x the rise time
    lambda_values             number of non-zero λ values (λ = 0 is always added)
    lambda_decades            the non-zero λ values span this many decades
    m_before, m_after         the m grid runs from m0 - m_before to m0 + m_after
    min_tail_bins             the cut always leaves at least this many tail bins
    clip_negative_irf         set negative IRF samples to zero, in both modes (prompt, section 3)
    near_best_tolerance       pairs within this fraction of the best deviance/dof are near-best
    lifetime_grid_points      points of the coarse lifetime grid before refinement
    curvature_block_ns        block length of the residual-curvature statistic (ns)
    """
    background_tail_fraction: float = 0.10
    window_margin_ns: float = 0.8
    rise_threshold: float = 0.02
    cut_smoothing_factor: float = 3.0
    lambda_top_factor: float = 1.5
    lambda_values: int = 13
    lambda_decades: float = 3.0
    m_before: int = 6
    m_after: int = 6
    min_tail_bins: int = 10
    clip_negative_irf: bool = True
    near_best_tolerance: float = 0.05
    lifetime_grid_points: int = 40
    curvature_block_ns: float = 1.0

    @classmethod
    def reference_grid(cls):
        """The settings that produced the regression numbers: m from m0 - 3 to m0 + 6."""
        return cls(m_before=3, m_after=6)

    def to_dict(self):
        """All settings as a plain dictionary (stored with every result)."""
        return asdict(self)
