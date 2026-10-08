"""Identify the model behind Transport-Optimizer's ER316 calibration files.

Reproduces calibration_{F3,F4}_10kHz_ER316.csv (trap minimum and curvature
frequency vs rotation) under a grid of model hypotheses and reports the RMS
deviation of each. Usage:
    python scripts/fit_er316_calibration.py /path/to/transport-optimizer
"""

import csv
import os
import itertools
import sys
from pathlib import Path

import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

STATES = {"F3": (3, -3), "F4": (4, -3)}


def load(directory, name):
    with open(Path(directory) / f"calibration_{name}_10kHz_ER316.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    frac = np.array([float(r["polarization_rotation_over_pi"]) for r in rows])
    x = np.array([float(r["minimum_position_continuous_um"]) for r in rows])
    f = np.array([float(r["trap_frequency_curvature_kHz"]) for r in rows])
    return frac, x, f


def evaluate(directory, every=10, **hypothesis):
    b_angle = np.deg2rad(hypothesis.pop("b_angle_deg"))
    b_mag = hypothesis.pop("b_gauss")
    hyperfine = hypothesis.pop("hyperfine_scalar")
    pair = LatticePair(**hypothesis)
    model = StatePotential(pair, b_mag * np.array([np.cos(b_angle), np.sin(b_angle), 0]),
                           include_hyperfine_scalar=hyperfine)
    out = {}
    for name, state in STATES.items():
        frac, x_ref, f_ref = load(directory, name)
        sel = slice(None, None, every)
        x, f = model.track(state, np.pi * frac[sel])
        out[name] = (x * 1e6 - x_ref[sel], f / 1e3 - f_ref[sel])
    return out


def main(directory):
    b0 = magic_field()
    grid = dict(
        handedness=[1, -1],
        rotation_sense=[1, -1],
        b_angle_deg=[45, 135, -45, -135],
        hyperfine_scalar=[True, False],
    )
    results = []
    for values in itertools.product(*grid.values()):
        hyp = dict(zip(grid, values))
        res = evaluate(directory, every=20, extinction_ratio=316.0, b_gauss=b0, **hyp)
        dx = np.sqrt(np.mean(np.concatenate([r[0] for r in res.values()]) ** 2)) * 1e3
        df = np.sqrt(np.mean(np.concatenate([r[1] for r in res.values()]) ** 2)) * 1e3
        results.append((df + dx, dx, df, hyp))
    results.sort(key=lambda r: r[0])
    print("rms position error (nm), rms frequency error (Hz), hypothesis")
    for _, dx, df, hyp in results[:8]:
        print(f"{dx:9.2f} {df:9.2f}  {hyp}")
    return results


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])  # run from anywhere
    main(sys.argv[1] if len(sys.argv) > 1 else "data/reference")
