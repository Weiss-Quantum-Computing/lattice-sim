"""Compare the cooling model with the measured post-cooling excitation per
axis (lab, 2026-10-13: 3.1 / 3.8 / 1.2 % on x / y / z with a 0.8 %
background, mostly spontaneous emission), under two hypotheses: a hotter
initial state, or dark-state scatter per optical-pumping pulse. Then
re-evaluates the protocol options under dark-state scatter.

Rabi-frequency sets: 'lab' (the manifest powers, 6.77 / 4.91 kHz for the
-2 / -1 pulse; the headline) and '20' (both pulses at 20 kHz, the value used
before 2026-10-15).

Needs data/psc_ladders.npz and data/psc_pulses.npz.
Writes data/psc_measured_comparison.json.
Usage: python scripts/psc_measured_comparison.py
"""

import json
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
SETS = ("lab", "20")
TEMPS_UK = (5, 6, 7, 8)
DARK = (0.005, 0.01, 0.015, 0.02, 0.03)
DARK_OPTIONS = (0.01, 0.015, 0.02)


def excitation(lib, op, seq, temperature_k=5e-6):
    h = psc.simulate_factorized(seq, lib, op, p0=psc.initial_state(temperature_k=temperature_k))
    return np.array([1 - h[-1][a][0] for a in psc.AXES]) * 100, psc.ground_3d_factorized(h)


def rms(ex, meas):
    return float(np.sqrt(np.mean((ex - meas) ** 2)))


if __name__ == "__main__":
    ladders = psc.load_ladders()
    tables = dict(np.load(Path("data") / "psc_pulses.npz"))
    meas = MEASURED_PCT - BACKGROUND_PCT
    print(f"measured, background-subtracted non-ground (%): {meas}; 3D ground {np.prod(1 - meas / 100):.4f}")
    lab = sequence(12, 6, 13, 12)
    out = dict(measured_pct=MEASURED_PCT.tolist(), background_pct=BACKGROUND_PCT, per_set={})
    for name in SETS:
        lib = psc.TableLibrary(ladders, tables, name)
        op0 = psc.optical_pumping_model(ladders)
        r = dict(temperature={}, dark={}, options={})
        print(f"--- set {name}, lab sequence")
        for t in TEMPS_UK:
            ex, g = excitation(lib, op0, lab, t * 1e-6)
            r["temperature"][str(t)] = dict(excitation_pct=ex.tolist(), ground_3d=g, rms_pct=rms(ex, meas))
            print(f"  T {t} uK, no dark scatter: {np.round(ex, 2)}  3D {g:.4f}  rms {rms(ex, meas):.2f}")
        for d in DARK:
            ex, g = excitation(lib, psc.optical_pumping_model(ladders, dark_prob=d), lab)
            r["dark"][str(d)] = dict(excitation_pct=ex.tolist(), ground_3d=g, rms_pct=rms(ex, meas))
            print(f"  5 uK, dark {100 * d:.1f} %/OP: {np.round(ex, 2)}  3D {g:.4f}  rms {rms(ex, meas):.2f}")
        # best single dark-scatter value on a fine grid (rms over the three axes)
        fine = np.round(np.arange(0.0, 0.0401, 0.0025), 4)
        fits = [(rms(excitation(lib, psc.optical_pumping_model(ladders, dark_prob=d), lab)[0], meas), d) for d in fine]
        e_best, d_best = min(fits)
        ex, g = excitation(lib, psc.optical_pumping_model(ladders, dark_prob=d_best), lab)
        r["dark_fit"] = dict(dark=float(d_best), rms_pct=e_best, excitation_pct=ex.tolist(), ground_3d=g)
        print(f"  best dark (rms over axes): {100 * d_best:.2f} %/OP -> {np.round(ex, 2)}, rms {e_best:.2f}")
        for d in DARK_OPTIONS:
            op = psc.optical_pumping_model(ladders, dark_prob=d)
            print(f"  options at dark {100 * d:.1f} %/OP:")
            o = {}
            for nm, seq in (("lab 30+10", lab), ("6/6 + 13/12, 30+10", sequence(6, 6, 13, 12)),
                            ("lab 40+10", sequence(12, 6, 13, 12, cycles=(40, 10))),
                            ("6/6 + 13/12, 40+10", sequence(6, 6, 13, 12, cycles=(40, 10))),
                            ("lab 60+10", sequence(12, 6, 13, 12, cycles=(60, 10)))):
                ex, g = excitation(lib, op, seq)
                o[nm] = dict(excitation_pct=ex.tolist(), ground_3d=g)
                print(f"    {nm:20s} {np.round(ex, 2)}  3D {g:.4f}")
            scan = [(excitation(lib, op, sequence(a, b, 13, 12))[1], a, b) for a in GRID for b in GRID]
            g, a, b = max(scan)
            o["best loop 1 (loop 2 at 13/12)"] = dict(t2a=float(a), t1a=float(b), ground_3d=g)
            print(f"    best loop 1 (loop 2 at 13/12): {a:g}/{b:g} deg, 3D {g:.4f}")
            r["options"][str(d)] = o
        out["per_set"][name] = r
    with open(Path("data/psc_measured_comparison.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print("wrote data/psc_measured_comparison.json")
