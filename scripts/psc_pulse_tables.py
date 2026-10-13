"""Pulse transfer matrices |U|^2 of the -2 and -1 AFP pulses on every axis
and every rotation 0.5-20 deg, for several carrier Rabi frequencies.

Variants: -2 pulse at HSR -6 kHz; -1 pulse at HSR -4 kHz and -6 kHz; all
centred at the programmed offsets (k x f_a above the theta = 0 carrier).
Needs data/psc_ladders.npz (scripts/psc_build_ladders.py).
Writes data/psc_pulses.npz (git-ignored, ~10 min).
Usage: python scripts/psc_pulse_tables.py [omega0_kHz ...]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import psc  # noqa: E402

OMEGA0_KHZ = (10.0, 14.0, 20.0, 40.0)
VARIANTS = ((2, -6e3), (1, -4e3), (1, -6e3))
OUT = Path("data") / "psc_pulses.npz"


def thetas(ladders):
    return sorted(k[1] for k in ladders if k[0] == "x" and isinstance(k[1], float) and k[1] > 0)


if __name__ == "__main__":
    omegas = [float(a) for a in sys.argv[1:]] or list(OMEGA0_KHZ)
    ladders = psc.load_ladders()
    arrays = dict(np.load(OUT)) if OUT.exists() else {}
    t0 = time.time()
    for om in omegas:
        for axis in psc.AXES:
            for th in thetas(ladders):
                lad = psc.ladder_at(ladders, axis, th)
                for kind, hsr in VARIANTS:
                    key = f"{om:g}_{axis}_{th:.1f}_{kind}_{hsr:g}"
                    if key not in arrays:
                        arrays[key] = psc.pulse_matrix(lad, kind, hsr, om * 1e3).astype(np.float32)
            print(f"Omega0 {om:g} kHz, {axis}: {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT}")
