"""Gaussian vs numerical-eigenstate overlap along x at the gate stop voltages.

Uses the model potential tables built with the measured extinction profile
(scripts/build_profile_tables.py, handedness -1) and the harmonic parameters
of the same model. Prints O_x(Gaussian) / O_x(numerical).
"""

import json

import numpy as np

from lattice_sim.gates.collision import oscillator_width
from lattice_sim.motion.transport1d import PotentialTable, Transport1D, harmonic_parameters

A = 4.8374e-6
STOPS = {3: (4300, 4430, 4460, 4610, 4785), 1: (3400, 3470, 3525, 3610)}


def main():
    out = {}
    for m, volts in STOPS.items():
        tl = PotentialTable.load(f"data/x_ERmeas133_h-1_10kHz_F3m+{m}.npz")
        tr = PotentialTable.load(f"data/x_ERmeas133_h-1_10kHz_F3m-{m}.npz")
        cl, fl = harmonic_parameters(tl)
        cr, fr = harmonic_parameters(tr)
        el, er_ = Transport1D(tl), Transport1D(tr)
        for v in volts:
            th = np.pi * v / 5400
            psi_l, _ = el.ground_state(th)
            psi_r, _ = er_.ground_state(th)
            x = np.linspace(-1e-6, A + 1e-6, 40001)
            nl = np.interp(x, tl.x, np.abs(psi_l) ** 2 / (tl.x[1] - tl.x[0]), left=0, right=0)
            nr = np.interp(x - A, tr.x, np.abs(psi_r) ** 2 / (tr.x[1] - tr.x[0]), left=0, right=0)
            numeric = np.trapezoid(nl * nr, x)
            k = int(np.argmin(np.abs(tl.thetas - th)))
            d = A - abs(cl[k]) - abs(cr[k])
            s2 = oscillator_width(fl[k]) ** 2 + oscillator_width(fr[k]) ** 2
            gauss = np.exp(-d**2 / s2) / np.sqrt(np.pi * s2)
            out[f"m{m}_{v}"] = dict(numeric=float(numeric), gaussian=float(gauss), ratio=float(gauss / numeric))
            print(f"|3,+/-{m}> {v} V: d {d*1e9:6.1f} nm  O_x Gaussian/numerical = {gauss/numeric:.3f}")
    with open("data/p4_overlap_check.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main()
