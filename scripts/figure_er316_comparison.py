"""Figure: first-principles x-pair potential vs the ER316 calibration files.

Usage: python scripts/figure_er316_comparison.py /path/to/transport-optimizer out.png
"""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fit_er316_calibration import load
from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.dpi": 300,
})
COLORS = {"F3": "#1f5fa8", "F4": "#c2452d"}
LABELS = {"F3": r"$|3,-3\rangle$", "F4": r"$|4,-3\rangle$"}


def main(directory, out):
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    model = StatePotential(LatticePair(extinction_ratio=316.0, handedness=-1), b,
                           include_hyperfine_scalar=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for name, state in (("F3", (3, -3)), ("F4", (4, -3))):
        frac, x_ref, f_ref = load(directory, name)
        sel = slice(None, None, 5)
        x, f = model.track(state, np.pi * frac[sel])
        c = COLORS[name]
        axes[0, 0].plot(frac[sel], x * 1e6, color=c, lw=1.0, label=f"{LABELS[name]} model")
        axes[0, 0].plot(frac[::12], x_ref[::12], "o", ms=2.5, mfc="none", mec=c, mew=0.6,
                        label=f"{LABELS[name]} calibration")
        axes[0, 1].plot(frac[sel], f / 1e3, color=c, lw=1.0)
        axes[0, 1].plot(frac[::12], f_ref[::12], "o", ms=2.5, mfc="none", mec=c, mew=0.6)
        axes[1, 0].plot(frac[sel], (x * 1e6 - x_ref[sel]) * 1e3, color=c, lw=1.0)
        axes[1, 1].plot(frac[sel], (f / 1e3 - f_ref[sel]) * 1e3, color=c, lw=1.0)
    axes[0, 0].set_ylabel(r"Trap minimum $x_0$ ($\mu$m)")
    axes[0, 1].set_ylabel(r"Trap frequency $\omega/2\pi$ (kHz)")
    axes[1, 0].set_ylabel(r"$\Delta x_0$ (nm)")
    axes[1, 1].set_ylabel(r"$\Delta\omega/2\pi$ (Hz)")
    for ax in axes[1]:
        ax.set_xlabel(r"Rotation angle $\theta/\pi$")
        ax.axhline(0, color="0.5", lw=0.5)
    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-0.2, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1)
    axes[0, 0].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
