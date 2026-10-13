"""Build the vibrational ladders (energies, Franck-Condon overlaps, carrier
shift) of |4,-4> and |3,-3> on every axis for rotations 0-20 deg in 0.5 deg
steps, at the full-depth trap frequencies set by the programmed -1 offsets.

Writes data/psc_ladders.npz (git-ignored cache, ~5 min).
Usage: python scripts/psc_build_ladders.py
"""

import os
import time
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import psc  # noqa: E402

THETAS_DEG = np.round(np.arange(0.0, 20.0001, 0.5), 3)

if __name__ == "__main__":
    t0 = time.time()
    ladders = psc.compute_ladders(THETAS_DEG)
    psc.save_ladders(ladders)
    for axis in psc.AXES:
        lad = ladders[(axis, 12.0)]
        print(f"{axis}: f_u = {lad.e_u[1]:.1f} Hz, displacement at 12 deg = {lad.displacement_m*1e9:.1f} nm, "
              f"carrier shift = {lad.carrier_shift_hz:.1f} Hz")
    print(f"done in {time.time() - t0:.0f} s -> {psc.CACHE}")
