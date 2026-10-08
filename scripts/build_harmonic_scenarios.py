"""Harmonic trap parameters x0(theta), f(theta) from the model, per scenario.

Each scenario is (state, extinction ratio, handedness) for the x pair at the
10 kHz lattice scale. Output: data/harm_<state>_ER<er>_h<h>.npz with arrays
theta, x0_m, f_hz. Usage: python scripts/build_harmonic_scenarios.py PAIR
  PAIR = pm3 | pm1
"""

import sys
import time

import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

PAIRS = {"pm3": [(3, -3), (3, 3)], "pm1": [(3, -1), (3, 1)]}
ERS = {"pm3": [316.0, 1000.0, np.inf], "pm1": [1e4, 3e4, np.inf]}


def theta_grid(n=481):
    # uniform, plus a 4x denser band around pi/2 where the minimum moves fastest
    base = np.linspace(0, np.pi, n)
    dense = np.linspace(0.4 * np.pi, 0.6 * np.pi, n // 2)
    return np.unique(np.concatenate([base, dense]))


def main(pair_name):
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    thetas = theta_grid()
    for state in PAIRS[pair_name]:
        for er in ERS[pair_name]:
            for h in ((-1, 1) if np.isfinite(er) else (-1,)):
                started = time.time()
                model = StatePotential(LatticePair(extinction_ratio=er, handedness=h), b)
                x0, f = model.track(state, thetas)
                tag = f"harm_F{state[0]}m{state[1]:+d}_ER{er:g}_h{h:+d}"
                np.savez(f"data/{tag}.npz", theta=thetas, x0_m=x0, f_hz=f)
                print(f"{tag}: x0(pi) {x0[-1]*1e6:+.3f} um, min f {f.min()/1e3:.2f} kHz "
                      f"({time.time()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])  # paths below are relative to the repo root
    main(sys.argv[1])
