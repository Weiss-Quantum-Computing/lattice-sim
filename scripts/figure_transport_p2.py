"""Figure for phase P2: transport on model potentials, x pair.

(a) rotation angle along the optimized ramp K06
(b) lab-frame ground-state population along the ramp
(c) trap minimum versus rotation angle, |m_F| = 3 vs 1
(d) trap depth at theta = pi/2 versus rotating-arm extinction ratio
Usage: python scripts/figure_transport_p2.py out.png
"""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.motion.transport1d import load_waveform

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.dpi": 300, "lines.linewidth": 1.0,
})
C = {"m-3": "#1f5fa8", "m+3": "#c2452d", "m-1": "#2a9d8f", "m+1": "#b5872a", "cal": "0.35"}
LBL = {"m-3": r"$|3,-3\rangle$", "m+3": r"$|3,+3\rangle$", "m-1": r"$|3,-1\rangle$", "m+1": r"$|3,+1\rangle$"}


def main(out):
    t, theta, volts = load_waveform("data/K06_5ms_capon_polyband.csv")
    k = np.load("data/transport_K06.npz")
    g = np.load("data/transport_gate_K06.npz")
    d = np.load("data/depth_vs_er.npz")
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6))

    a = ax[0, 0]
    a.plot(t / 1e3, theta / np.pi, color="0.15")
    a.axhspan(3800 / 5400, 1.0, color="0.9", lw=0)
    a.set_xlabel("Time (ms)"), a.set_ylabel(r"Rotation angle $\theta/\pi$")
    a.text(0.2, 0.86, "stop-anywhere band\n(V ≥ 3.8 kV)", fontsize=6, color="0.35")

    a = ax[0, 1]
    a.plot(k["calF3_t"] / 1e3, 1 - k["calF3_p0"], color=C["cal"], ls="--",
           label=r"$|3,-3\rangle$, calibration (exact)")
    a.plot(k["x_ER316_10kHz_F3m-3_t"] / 1e3, 1 - k["x_ER316_10kHz_F3m-3_p0"], color=C["m-3"],
           label=r"$|3,-3\rangle$, model")
    a.plot(g["x_ER316_10kHz_F3m+3_t"] / 1e3, 1 - g["x_ER316_10kHz_F3m+3_p0"], color=C["m+3"],
           label=r"$|3,+3\rangle$, model")
    a.plot(g["x_ER316_10kHz_F3m+1_t"] / 1e3, 1 - g["x_ER316_10kHz_F3m+1_p0"], color=C["m+1"],
           label=r"$|3,+1\rangle$, model")
    a.set_yscale("log"), a.set_ylim(1e-7, 1)
    a.set_xlabel("Time (ms)"), a.set_ylabel(r"Excitation $1-P_0$")
    a.legend(loc="lower center", fontsize=5.5, ncol=1)

    a = ax[1, 0]
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    th = np.linspace(0, 0.995 * np.pi, 120)
    model = StatePotential(LatticePair(), b)
    for key, state in (("m-3", (3, -3)), ("m-1", (3, -1))):
        x, _ = model.track(state, th)
        a.plot(th / np.pi, -x * 1e6, color=C[key], label=LBL[key])
    a.set_xlabel(r"Rotation angle $\theta/\pi$"), a.set_ylabel(r"Displacement $|x_0|$ ($\mu$m)")
    a.legend(loc="upper left", fontsize=6)
    a.text(0.55, 0.3, "pure linear\npolarization", fontsize=6, color="0.35")

    a = ax[1, 1]
    er = d["er"]
    for key in ("m-3", "m+3", "m-1", "m+1"):
        a.plot(er, d[f"h-1_F3{key}_depth"] / 1e3, color=C[key], marker="o", ms=2.2, label=LBL[key])
    a.axvline(316, color="0.5", lw=0.5, ls=":")
    a.text(290, 300, "ER = 316", fontsize=6, color="0.35", ha="right")
    a.set_xscale("log"), a.set_ylim(0, 400)
    a.set_xlabel("Rotating-arm extinction ratio"), a.set_ylabel(r"Depth at $\theta=\pi/2$ (kHz)")
    a.legend(loc="lower right", fontsize=6, ncol=2)

    for axis, letter in zip(ax.flat, "abcd"):
        axis.text(-0.2, 1.02, letter, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
