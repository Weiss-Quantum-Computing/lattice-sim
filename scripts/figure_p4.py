"""Figure for phase 4: controlled collisions on the x axis.

(a) wave-packet separation vs EOM voltage
(b, c) collision energy and lattice-energy gaps to spin-exchange channels
(d) collisional phase for a ramp stopped at V (no hold)
(e) leakage out of |L:+m, R:-m> for the same ramps (DIS channel lengths)
(f) Gaussian / numerical-eigenstate overlap ratio at the stop voltages
Usage: python scripts/figure_p4.py out.png
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from lattice_sim.constants import A0
from lattice_sim.gates.collision import (MU, dis_channel_lengths, interaction_hz,
                                         scattering_operator, uniform_lengths)
from lattice_sim.viz.style import COLORS, apply

V_PI = 5400.0


def main(out):
    apply()
    r = json.load(open("data/p4_results.json", encoding="utf-8"))
    chk = json.load(open("data/p4_overlap_check.json", encoding="utf-8"))
    ops = {"DIS": scattering_operator(dis_channel_lengths()),
           "uniform 2800 a0": scattering_operator(uniform_lengths(2800.0))}
    col = {3: COLORS["blue"], 1: COLORS["teal"]}
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.6))

    a = ax[0, 0]
    for m in (3, 1):
        c = r[f"curves_m{m}_h-1"]
        v = np.array(c["theta"]) / np.pi * V_PI
        sel = v > 2500
        a.plot(v[sel], np.array(c["separation_nm"])[sel], color=col[m], label=rf"$|3,\pm{m}\rangle$")
    a.set_xlabel("EOM voltage (V)"), a.set_ylabel("Packet separation (nm)")
    a.set_ylim(0, 1500), a.legend(loc="upper right", fontsize=6)

    for m, axis, vmin in ((3, ax[0, 1], 3600), (1, ax[0, 2], 2900)):
        c = r[f"curves_m{m}_h-1"]
        v = np.array(c["theta"]) / np.pi * V_PI
        sel = v >= vmin
        o = np.array(c["overlap"])
        i0 = list(MU).index(m)
        for name, ls in (("DIS", "-"), ("uniform 2800 a0", "--")):
            u = interaction_hz(1.0, o) * np.real(ops[name][i0, i0])
            axis.plot(v[sel], u[sel] / 1e3, color="k", ls=ls, lw=1.0,
                      label=f"U, {'DIS' if name == 'DIS' else '2800 a$_0$'}")
        lat = np.array(c["lattice_hz"]) + np.array(c["zeeman_hz"])[None, :]
        shades = [COLORS["red"], COLORS["ochre"], COLORS["purple"], COLORS["grey"]]
        chans = [mu for mu in (m - 1, m - 2, 0, -m) if mu != m and abs(mu) <= 3]
        for mu, cc in zip(dict.fromkeys(chans), shades):
            j = list(MU).index(mu)
            axis.plot(v[sel], np.abs(lat[sel, j]) / 1e3, color=cc, lw=0.9,
                      label=rf"gap, $|{mu:+d},{-mu:+d}\rangle$")
        axis.set_yscale("log"), axis.set_ylim(0.1, 5e3)
        axis.set_xlabel("EOM voltage (V)"), axis.set_ylabel("Energy / h (kHz)")
        axis.set_title(rf"$|3,\pm{m}\rangle$", fontsize=7)
        axis.legend(fontsize=5, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.5)

    a, b = ax[1, 0], ax[1, 1]
    for m in (3, 1):
        for name, ls in (("DIS", "-"), ("uniform 2800 a0", "--")):
            s = r[f"scan_m{m}_h-1_{name}"]
            v = np.array(s["theta"]) / np.pi * V_PI
            a.plot(v, np.array(s["phi"]) / np.pi, color=col[m], ls=ls,
                   label=rf"$|3,\pm{m}\rangle$, {'DIS' if name == 'DIS' else '2800 a$_0$'}")
            if name == "DIS":
                b.plot(v, np.clip(s["leak"], 1e-8, None), color=col[m], label=rf"$|3,\pm{m}\rangle$")
    for y in (0.5, 1.0):
        a.axhline(y, color="0.6", lw=0.5, ls=":")
    a.set_ylim(0, 3), a.set_xlabel("Stop voltage (V)"), a.set_ylabel(r"Collisional phase ($\pi$)")
    a.legend(fontsize=5, loc="upper left")
    b.set_yscale("log"), b.set_ylim(1e-7, 1)
    b.set_xlabel("Stop voltage (V)"), b.set_ylabel("Spin-exchange leakage (DIS)")
    b.legend(fontsize=6, loc="upper left")

    a = ax[1, 2]
    for m in (3, 1):
        keys = sorted([k for k in chk if k.startswith(f"m{m}_")], key=lambda k: int(k.split("_")[1]))
        a.plot([int(k.split("_")[1]) for k in keys], [chk[k]["ratio"] for k in keys], "o-", ms=3,
               color=col[m], label=rf"$|3,\pm{m}\rangle$")
    a.axhline(1.0, color="0.6", lw=0.5)
    a.set_ylim(0.97, 1.02), a.set_xlabel("Stop voltage (V)")
    a.set_ylabel(r"$O_x$ Gaussian / numerical")
    a.legend(fontsize=6, loc="lower right")

    for axis, letter in zip(ax.flat, "abcdef"):
        axis.text(-0.28, 1.04, letter, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main(sys.argv[1] if len(sys.argv) > 1 else "figures/p4_collision.png")
