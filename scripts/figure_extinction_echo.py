"""Figure: measured rotating-arm extinction ratio and its echo-phase cost.

(a) ER vs rotation angle, scan 133-ms-SE-offset-neg2V-fine-2 (dip method)
(b) ER-independent differential shift between transport partners
(c) differential shift per unit ellipticity
(d) cumulative differential phase along the measured leg-1 trajectory
Usage: python scripts/figure_extinction_echo.py ER_CSV out.png
"""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lattice_sim.fields.extinction import read_rampol_er

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.dpi": 300, "lines.linewidth": 1.0,
})
LEG = {(1, "away"): ("#1f5fa8", "o"), (1, "back"): ("#1f5fa8", "s"),
       (2, "away"): ("#c2452d", "o"), (2, "back"): ("#c2452d", "s")}
PAIR = {3: ("#1f5fa8", r"$|3,\pm3\rangle$"), 1: ("#2a9d8f", r"$|3,\pm1\rangle$")}


def main(path, out):
    rows = [r for r in read_rampol_er(path) if r["method"] == "dip"]
    d = np.load("data/echo_sensitivity.npz")
    th = np.rad2deg(d["theta"])
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.8))

    a = ax[0, 0]
    for (leg, direction), (c, mk) in LEG.items():
        pts = [r for r in rows if int(r["leg"]) == leg and r["direction"] == direction]
        x = np.abs([r["rotation_deg"] for r in pts]) + (0.8 if leg == 2 else -0.8)
        y = np.array([r["er"] for r in pts])
        lo = np.array([r["lower_bound"] == 1 for r in pts])
        err = np.array([[r["er_sigma_lo"], r["er_sigma_hi"]] for r in pts]).T
        a.errorbar(x[~lo], y[~lo], yerr=np.nan_to_num(err[:, ~lo]), fmt=mk, ms=2.8, color=c,
                   mfc=c if direction == "away" else "white", mew=0.7, elinewidth=0.6, capsize=0,
                   label=f"leg {leg}, {direction}")
        a.plot(x[lo], y[lo], "^", ms=3.2, color=c, mfc="none", mew=0.6)
    a.set_yscale("log"), a.set_xlim(-5, 185), a.set_xticks(range(0, 181, 45))
    a.set_xlabel("Rotation angle (deg)"), a.set_ylabel("Extinction ratio")
    a.text(2, 5.5e4, "△ lower bound", fontsize=6, color="0.35")
    a.legend(loc="lower left", fontsize=5.5, ncol=2)

    a = ax[0, 1]
    for m, (c, lbl) in PAIR.items():
        a.plot(th, d[f"E0_m{m}"] / 1e3, color=c, label=lbl)
    a.axhline(0, color="0.6", lw=0.5)
    a.set_xlim(0, 180), a.set_xticks(range(0, 181, 45))
    a.set_xlabel("Rotation angle (deg)"), a.set_ylabel(r"$\Delta E_{\eta=0}/h$ (kHz)")
    a.legend(loc="lower right", fontsize=6)

    a = ax[1, 0]
    for m, (c, lbl) in PAIR.items():
        a.plot(th, d[f"S_m{m}"] / 1e3 * 1e-3, color=c, label=lbl)
    a.axhline(0, color="0.6", lw=0.5)
    a.set_xlim(0, 180), a.set_xticks(range(0, 181, 45))
    a.set_xlabel("Rotation angle (deg)"), a.set_ylabel(r"$\partial\Delta E/\partial\eta$ (kHz per $10^{-3}$)")
    a.legend(loc="lower right", fontsize=6)

    a = ax[1, 1]
    t = d["traj_t"] * 1e3
    for m, (c, lbl) in PAIR.items():
        a.plot(t, -d[f"cum_drag_m{m}"], color=c, ls="--")
        a.plot(t, -(d[f"cum_drag_m{m}"] + d[f"cum_ell_m{m}"]), color=c, label=lbl)
    a.set_xlabel("Time in transport (ms)"), a.set_ylabel(r"Cumulative $|\phi|$ (rad)")
    a.text(0.36, 0.88, "dashed: η = 0\nsolid: + measured η (upper bound)", transform=a.transAxes,
           fontsize=5.5, color="0.35", va="top")
    a.legend(loc="upper left", fontsize=6)
    a2 = a.twinx()
    a2.plot(t, np.rad2deg(d["traj_theta"]), color="0.75", lw=0.6, zorder=0)
    a2.set_ylabel("Rotation (deg)", color="0.55"), a2.tick_params(axis="y", colors="0.55")
    a2.set_ylim(0, 400), a2.set_yticks([0, 90, 180])

    for axis, letter in zip(ax.flat, "abcd"):
        axis.text(-0.2, 1.02, letter, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
