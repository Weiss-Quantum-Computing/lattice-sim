"""Tabulate model potentials V(x, theta) for the x pair, one file per state.

Usage: python scripts/build_model_tables.py [extinction_ratio] [n_theta]
"""

import sys
import time

import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.motion.transport1d import PotentialTable, grid_for_direction

STATES = [(3, -3), (4, -3), (3, 3), (3, -1), (3, 1)]


def main(er=316.0, n_theta=2881, trap_hz=10e3, out_dir="data", states=None):
    states = STATES if states is None else states
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    pair = LatticePair(extinction_ratio=er, handedness=-1)
    model = StatePotential(pair, b, trap_frequency_hz=trap_hz, include_hyperfine_scalar=True)
    thetas = np.linspace(0.0, np.pi, n_theta)
    for state in states:
        # Direction from the first 45% of the rotation, where every state's
        # minimum is still continuous (|3,+/-1> can lose its trap near pi/2)
        x_early, _ = model.track(state, np.linspace(0, 0.45 * np.pi, 19))
        sign = np.sign(x_early[-1])
        x_end = [sign * pair.spacing_m / 2]
        x = grid_for_direction(sign)
        started = time.time()
        table = PotentialTable.from_model(model, state, thetas, x)
        tag = f"x_ER{er:g}_{trap_hz/1e3:g}kHz_F{state[0]}m{state[1]:+d}"
        table.save(f"{out_dir}/{tag}.npz")
        print(f"{state}: moves {'+' if sign > 0 else '-'}, end {x_end[-1]*1e6:+.3f} um, "
              f"{time.time()-started:.0f}s -> {tag}.npz", flush=True)


if __name__ == "__main__":
    er = float(sys.argv[1]) if len(sys.argv) > 1 else 316.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2881
    main(er, n)
