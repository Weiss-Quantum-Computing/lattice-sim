"""Score transport ramps for one or more internal states.

Engines:
  gaussian  exact solution for the harmonic calibration (Transport-Optimizer CSVs)
  split     split-step propagation on a tabulated potential (model or harmonic)

Usage: python scripts/transport_compare.py RAMP.csv OUT.npz STATE_TAG [STATE_TAG ...]
  STATE_TAG = calF3 | calF4 | <table file stem in data/>
"""

import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

from fit_er316_calibration import load
from lattice_sim.motion.gaussian_exact import propagate_gaussian
from lattice_sim.motion.transport1d import PotentialTable, Transport1D, band_mask, load_waveform

OPT_DIR = "/home/claude/transport-optimizer"


def run(ramp, tag, dt_us=0.25, sample_us=2.5):
    t, theta, volts = load_waveform(ramp)
    started = time.time()
    if tag.startswith("cal"):
        frac, xc, fc = load(OPT_DIR, tag[3:])
        cx = CubicSpline(np.pi * frac, np.abs(xc) * 1e-6)
        cf = CubicSpline(np.pi * frac, fc * 1e3)
        ts = np.arange(t[0], t[-1] + 1e-9, sample_us)
        ts, p0, lag = propagate_gaussian(t, theta, cx, cf, sample_times_us=ts)
        engine = "gaussian"
    else:
        table = PotentialTable.load(f"data/{tag}.npz")
        ts, p0, mean_x = Transport1D(table).propagate(t, theta, dt_us=dt_us, sample_every_us=sample_us)
        engine = "split"
    v = np.interp(ts, t, volts)
    m = band_mask(v)
    print(f"{tag:24s} {engine:8s} band min P0 {p0[m].min():.6f}  path min {p0.min():.6f}  "
          f"final {p0[-1]:.6f}  ({time.time() - started:.0f}s)", flush=True)
    return ts, v, p0


def main(ramp, out, tags):
    results = {}
    for tag in tags:
        ts, v, p0 = run(ramp, tag)
        results[f"{tag}_t"], results[f"{tag}_v"], results[f"{tag}_p0"] = ts, v, p0
    np.savez(out, **results)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
