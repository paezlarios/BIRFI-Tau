"""
Write the example data set used by walkthrough.ipynb: simulated, so every true value is known.

calibration/  cal_01 ... cal_05: a monoexponential dye, τ = 1.87 ns, 10^4 counts peak,
              background 5 counts/bin. cal_04 is deliberately a different sample
              (τ = 2.20 ns) to show that the calibration detects and excludes it.
samples/      sample_01 ... sample_12: two components, τ1 = 0.8 ns and τ2 = 3.2 ns, amplitude
              fractions 60 % and 40 %, 2 x 10^4 counts peak, background 8 counts/bin, and a
              timing drift growing linearly from 0 to 0.4 bins (0.04 ns) across the series.
Binning: 250 bins of 100 ps (one 40 MHz period), like the reference decay.
Run:  python make_example_data.py
"""
from pathlib import Path

import numpy as np

from tcspc_lifetime.simulate import expected_sample, poisson_decay, smooth_irf

TIME_NS = np.arange(250) * 0.1
HERE = Path(__file__).parent / "data"


def write(path, counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([TIME_NS, counts]), delimiter=",", fmt=["%.4f", "%d"],
               header="time (ns), counts")


for k in range(1, 6):
    tau = 2.20 if k == 4 else 1.87
    expected, _ = expected_sample(smooth_irf(TIME_NS), TIME_NS, [tau], [1.0], 1e4, 5.0)
    write(HERE / "calibration" / f"cal_{k:02d}.csv", poisson_decay(expected, seed=500 + k))

for k in range(1, 13):
    drift_ns = 0.04 * (k - 1) / 11
    expected, _ = expected_sample(smooth_irf(TIME_NS, drift_ns), TIME_NS, [0.8, 3.2], [0.6, 0.4], 2e4, 8.0)
    write(HERE / "samples" / f"sample_{k:02d}.csv", poisson_decay(expected, seed=600 + k))

print(f"wrote {len(list(HERE.rglob('*.csv')))} files to {HERE}")
