"""Lattice depth and trap frequency of the transport pairs at theta = pi/2
as a function of the rotating-arm extinction ratio (x pair, 10 kHz scale).

Writes data/depth_vs_er.npz; usage: python scripts/depth_vs_extinction.py
"""

import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.constants import H, MASS_CS
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

PAIRS = {"pm3": [(3, -3), (3, 3)], "pm1": [(3, -1), (3, 1)]}


def site_properties(v, x):
    """Peak-to-peak modulation over one period (the trap depth) and the
    harmonic frequency at the global minimum. At theta = pi/2 each state has
    one well per lattice period."""
    i = int(np.argmin(v))
    depth = v.max() - v[i]
    k = slice(max(i - 3, 0), i + 4)
    c2 = np.polyfit(x[k] - x[i], v[k], 2)[0]
    freq = np.sqrt(max(2 * c2 * H, 0) / MASS_CS) / (2 * np.pi)
    return depth, freq, x[i]


def main():
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    ers = np.logspace(2, 5, 25)
    a = LatticePair().spacing_m
    x = np.linspace(-a / 2, a / 2, 2001, endpoint=False)
    out = {"er": ers}
    for handed in (1, -1):
        for name, states in PAIRS.items():
            for state in states:
                depth, freq = [], []
                for er in ers:
                    model = StatePotential(LatticePair(extinction_ratio=er, handedness=handed), b)
                    v = model.potential(state, x, np.pi / 2)
                    d, f, _ = site_properties(v, x)
                    depth.append(d), freq.append(f)
                key = f"h{handed:+d}_F{state[0]}m{state[1]:+d}"
                out[key + "_depth"] = np.array(depth)
                out[key + "_freq"] = np.array(freq)
                print(key, "ER 316: depth %.1f kHz, f %.2f kHz; ER 1e4: depth %.1f kHz, f %.2f kHz"
                      % (np.interp(316, ers, depth) / 1e3, np.interp(316, ers, freq) / 1e3,
                         np.interp(1e4, ers, depth) / 1e3, np.interp(1e4, ers, freq) / 1e3), flush=True)
    np.savez("data/depth_vs_er.npz", **out)


if __name__ == "__main__":
    main()
