"""Figure: robust transport ramps and the handedness discriminator.

(a) theta(t) for K06 and the robust ramps
(b) band-minimum excitation per scenario, |3,+/-3>: K06 vs R3
(c) the same for |3,+/-1>: R1 at 5 ms and 8 ms
(d) f(|4,-4>) / f(|3,-3>) at theta = pi/2 vs extinction ratio, both handedness
Usage: python scripts/figure_ramps_p2b.py out.png
"""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from design_robust_ramp import scenarios
from depth_vs_extinction import site_properties
from lattice_sim.atom.ground_state import magic_field
from lattice_sim.control.ramp_optimizer import RampProblem
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.motion.transport1d import load_waveform

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.dpi": 300, "lines.linewidth": 1.0,
})
COL = {"K06": "0.35", "R3": "#1f5fa8", "R1_5": "#2a9d8f", "R1_8": "#b5872a", "h-1": "#c2452d", "h+1": "#1f5fa8"}


def short(name):
    s = name.replace("harm_F3m", "").replace("_ER", ", ER ").replace("_h", ", h ")
    return s.replace("_f0.95", ", f×0.95").replace("_f1.05", ", f×1.05").replace("ERinf", "ER ∞").replace("ER inf", "ER ∞")


def scores(pair, ramp):
    sc = scenarios(pair)
    t, th, v = load_waveform(ramp)
    prob = RampProblem(sc, duration_us=t[-1])
    bm, _ = prob.score_waveform(t, th, v)
    return [short(s.name) for s in sc], 1 - bm


def dot_panel(ax, names, series):
    y = np.arange(len(names))
    for (label, key, vals), dy in zip(series, (-0.15, 0.15)):
        ax.plot(vals, y + dy, "o", ms=3, color=COL[key], label=label)
    ax.set_yticks(y, names, fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Worst stop-anywhere excitation $1-P_0$")
    ax.invert_yaxis()
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)


def main(out):
    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    ax_a, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for key, path, label in (("K06", "data/K06_5ms_capon_polyband.csv", "K06 (current)"),
                             ("R3", "data/R3_5ms_robust.csv", r"R3, $|3,\pm3\rangle$, 5 ms"),
                             ("R1_5", "data/R1_5ms_robust.csv", r"R1, $|3,\pm1\rangle$, 5 ms"),
                             ("R1_8", "data/R1_8ms_robust.csv", r"R1, $|3,\pm1\rangle$, 8 ms")):
        t, th, _ = load_waveform(path)
        ax_a.plot(t / 1e3, th / np.pi, color=COL[key], label=label, ls="--" if key == "K06" else "-")
    ax_a.set_xlabel("Time (ms)"), ax_a.set_ylabel(r"Rotation angle $\theta/\pi$")
    ax_a.legend(loc="lower right", fontsize=6)

    names, k06 = scores("pm3", "data/K06_5ms_capon_polyband.csv")
    _, r3 = scores("pm3", "data/R3_5ms_robust.csv")
    dot_panel(ax_b, names, [("K06", "K06", k06), ("R3", "R3", r3)])
    names, r15 = scores("pm1", "data/R1_5ms_robust.csv")
    _, r18 = scores("pm1", "data/R1_8ms_robust.csv")
    dot_panel(ax_c, names, [("R1, 5 ms", "R1_5", r15), ("R1, 8 ms", "R1_8", r18)])
    ax_c.set_xticks([2e-4, 5e-4, 1e-3, 2e-3], [r"$2\times10^{-4}$", r"$5\times10^{-4}$", r"$10^{-3}$", r"$2\times10^{-3}$"])

    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    a = LatticePair().spacing_m
    x = np.linspace(-a / 2, a / 2, 2001, endpoint=False)
    ers = np.logspace(2, 5, 13)
    for h in (-1, 1):
        ratio = []
        for er in ers:
            m = StatePotential(LatticePair(extinction_ratio=er, handedness=h), b)
            f44 = site_properties(m.potential((4, -4), x, np.pi / 2), x)[1]
            f33 = site_properties(m.potential((3, -3), x, np.pi / 2), x)[1]
            ratio.append(f44 / f33)
        ax_d.plot(ers, ratio, "o-", ms=2.2, color=COL[f"h{h:+d}"], label=f"handedness {h:+d}")
    m = StatePotential(LatticePair(), b)
    r_inf = (site_properties(m.potential((4, -4), x, np.pi / 2), x)[1]
             / site_properties(m.potential((3, -3), x, np.pi / 2), x)[1])
    ax_d.axhline(r_inf, color="0.5", lw=0.5, ls=":")
    ax_d.text(1.2e2, r_inf + 0.03, "linear polarization", fontsize=6, color="0.35")
    ax_d.set_xscale("log")
    ax_d.set_xlabel("Rotating-arm extinction ratio")
    ax_d.set_ylabel(r"$f_{|4,-4\rangle}/f_{|3,-3\rangle}$ at $\theta=\pi/2$")
    ax_d.legend(loc="upper right", fontsize=6)

    for axis, letter in zip((ax_a, ax_d, ax_b, ax_c), "adbc"):
        axis.text(-0.16 if letter in "ad" else -0.42, 1.02, letter, transform=axis.transAxes,
                  fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])  # paths below are relative to the repo root
    main(sys.argv[1])
