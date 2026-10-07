"""Transport P0 of a state versus rotating-arm extinction ratio (x pair).

Usage: python scripts/er_scan_transport.py F M ER RAMP.csv
Builds a two-site grid table (both neighbouring sites inside) and reports the
final population in the ground state of the intended site (x0 = sign * a/2)
and of the opposite one.
"""

import sys

import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.motion.transport1d import PotentialTable, Transport1D, load_waveform, uniform_grid


def site_ground_state(x, v, centre, half_width=0.8e-6):
    masked = np.where(np.abs(x - centre) < half_width, v, v.max())
    table = PotentialTable(np.array([0.0, 1.0]), x, np.vstack([masked, masked]))
    return Transport1D(table).ground_state(0.0)[0]


def main(f, m, er, ramp, handed=-1, n_theta=2881):
    state = (int(f), int(m))
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    pair = LatticePair(extinction_ratio=er, handedness=handed)
    model = StatePotential(pair, b)
    x_early, _ = model.track(state, np.linspace(0, 0.4 * np.pi, 9))
    sign = np.sign(x_early[-1])
    x = uniform_grid(-3.6e-6, 3.6e-6, 1792)
    table = PotentialTable.from_model(model, state, np.linspace(0, np.pi, n_theta), x)
    t, theta, _ = load_waveform(ramp)
    tr = Transport1D(table)
    _, _, _, psis = tr.propagate(t, theta, dt_us=0.25, sample_every_us=t[-1], keep_states=True)
    psi = psis[-1]
    a = pair.spacing_m
    v_end = table.at(np.pi)
    p_target = abs(np.vdot(site_ground_state(x, v_end, sign * a / 2), psi)) ** 2
    p_wrong = abs(np.vdot(site_ground_state(x, v_end, -sign * a / 2), psi)) ** 2
    p_side = float(np.sum(np.abs(psi[np.sign(x) == sign]) ** 2))
    print(f"{state} ER {er:g} h {handed:+d}: P0(target site) {p_target:.6f}  "
          f"P0(opposite site) {p_wrong:.6f}  P(target side) {p_side:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])
