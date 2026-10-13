"""Compare the cooling model with the measured post-cooling excitation per
axis (lab, 2026-10-13: 3.1 / 3.8 / 1.2 % on x / y / z with a 0.8 %
background, mostly spontaneous emission), under two hypotheses: a hotter
initial state, or dark-state scatter per optical-pumping pulse. Then
re-evaluates the protocol options under the fitted dark-state scatter.

Needs data/psc_ladders.npz and data/psc_pulses.npz.
Usage: python scripts/psc_measured_comparison.py
"""

import os
import sys
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path("scripts").resolve()))

from lattice_sim.cooling import psc  # noqa: E402
from psc_analysis import GRID, sequence  # noqa: E402

MEASURED_PCT = np.array([3.1, 3.8, 1.2])
BACKGROUND_PCT = 0.8


def excitation(lib, op, seq, temperature_k=5e-6):
    h = psc.simulate_factorized(seq, lib, op, p0=psc.initial_state(temperature_k=temperature_k))
    return np.array([1 - h[-1][a][0] for a in psc.AXES]) * 100, psc.ground_3d_factorized(h)


if __name__ == "__main__":
    ladders = psc.load_ladders()
    tables = dict(np.load(Path("data") / "psc_pulses.npz"))
    meas = MEASURED_PCT - BACKGROUND_PCT
    print(f"measured, background-subtracted non-ground (%): {meas}; 3D ground {np.prod(1 - meas / 100):.4f}")
    lab = sequence(12, 6, 13, 12)
    for om in (20.0, 10.0):
        lib = psc.TableLibrary(ladders, tables, om)
        op0 = psc.optical_pumping_model(ladders)
        print(f"--- Omega0 {om:g} kHz, lab sequence")
        for t in (5, 6, 7, 8):
            ex, g = excitation(lib, op0, lab, t * 1e-6)
            print(f"  T {t} uK, no dark scatter: {np.round(ex, 2)}  3D {g:.4f}")
        for d in (0.005, 0.01, 0.015, 0.02, 0.03):
            ex, g = excitation(lib, psc.optical_pumping_model(ladders, dark_prob=d), lab)
            print(f"  5 uK, dark {100 * d:.1f} %/OP: {np.round(ex, 2)}  3D {g:.4f}")
        for d in (0.015, 0.02):
            op = psc.optical_pumping_model(ladders, dark_prob=d)
            print(f"  options at dark {100 * d:.1f} %/OP:")
            for name, seq in (("lab 30+10", lab), ("6/6 + 13/12, 30+10", sequence(6, 6, 13, 12)),
                              ("lab 40+10", sequence(12, 6, 13, 12, cycles=(40, 10))),
                              ("6/6 + 13/12, 40+10", sequence(6, 6, 13, 12, cycles=(40, 10))),
                              ("lab 60+10", sequence(12, 6, 13, 12, cycles=(60, 10)))):
                ex, g = excitation(lib, op, seq)
                print(f"    {name:20s} {np.round(ex, 2)}  3D {g:.4f}")
            scan = [(excitation(lib, op, sequence(a, b, 13, 12))[1], a, b) for a in GRID for b in GRID]
            g, a, b = max(scan)
            print(f"    best loop 1 (loop 2 at 13/12): {a:g}/{b:g} deg, 3D {g:.4f}")
