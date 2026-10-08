"""Figure for phase 3: echo phase and contrast budget.

(a) residual echo phase from the EOM leg transient vs leg separation
(b) contrast budget by source at 150 ms separation (tau = 50 ms)
(c) pi-train contrast vs number of pulses (worst input state)
(d) Du's |3,+/-1> contrast data vs the two leg-to-leg mechanisms
Usage: python scripts/figure_p3.py out.png
"""

import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lattice_sim.spin.pulses import worst_input_contrast

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.dpi": 300, "lines.linewidth": 1.0,
})
PAIR = {3: "#1f5fa8", 1: "#2a9d8f"}
SRC = [("c_maps", "map pulses", "#6b6b6b"), ("c_scatter", "scattering", "#b5872a"),
       ("c_field", "field noise", "#1f5fa8"), ("c_leg_drift", "leg intensity drift", "#c2452d"),
       ("c_pulses", "π train", "#7a5195"), ("c_motion", "motional excitation", "#2a9d8f"),
       ("c_inhomogeneous", "transient × envelope", "#e07b39"), ("c_intensity", "RIN", "#a0a0a0")]


def main(out):
    res = json.load(open("data/p3_results.json"))
    pred = res["predictions"]
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.9))

    a = ax[0, 0]
    for m in (3, 1):
        for tau, ls in ((20, ":"), (50, "-"), (100, "--")):
            rows = sorted([r for r in pred if r["m"] == m and r["axes"] == "x" and r["tau_ms"] == tau],
                          key=lambda r: r["actual_separation_ms"])
            x = [r["actual_separation_ms"] for r in rows]
            a.plot(x, [abs(r["phase_rad"]) for r in rows], ls=ls, marker="o", ms=2, color=PAIR[m],
                   label=(rf"$|3,\pm{m}\rangle$, " if tau == 50 else "") + f"τ = {tau} ms")
    a.axhline(0.1, color="0.5", lw=0.5, ls=":")
    a.set_yscale("log"), a.set_ylim(1e-4, 5)
    a.set_xlabel("Leg separation, start to start (ms)"), a.set_ylabel("|Residual echo phase| (rad)")
    a.legend(fontsize=5.5, ncol=2, loc="lower left")

    a = ax[0, 1]
    configs = [(3, "x"), (3, "xyz"), (1, "x"), (1, "xyz")]
    labels = []
    for i, (m, axes) in enumerate(configs):
        r = [r for r in pred if r["m"] == m and r["axes"] == axes and r["tau_ms"] == 50
             and abs(r["separation_ms"] - 150) < 1][0]
        left = 0.0
        for key, name, col in SRC:
            w = -np.log(max(r[key], 1e-12))
            a.barh(i, w, left=left, color=col, height=0.6, label=name if i == 0 else None,
                   edgecolor="white", linewidth=0.5)
            left += w
        a.text(left + 0.003, i, f"C = {r['contrast']:.3f}", va="center", fontsize=6)
        labels.append(rf"$|3,\pm{m}\rangle$, {axes}")
    a.set_yticks(range(len(configs)), labels), a.invert_yaxis()
    a.set_xlabel(r"Contrast loss $-\ln C$ (150 ms separation)")
    a.set_xlim(0, 0.27)
    a.legend(fontsize=5.2, ncol=1, loc="center right")

    a = ax[1, 0]
    n = np.array([1, 3, 5, 9, 13, 17, 25])
    for kind, col, ls in (("CP (pi_x)", "#c2452d", "-"), ("CPMG (pi_y)", "#b5872a", "--"), ("XY-8", "#1f5fa8", "-")):
        c = [worst_input_contrast(kind, k, eps_sigma=0.01, detuning_over_rabi=0.01, n_shots=120) for k in n]
        a.plot(n, 1 - np.array(c), "o", ls=ls, ms=2.5 if ls == "-" else 4, mfc="none" if ls == "--" else col,
               color=col, label=kind.replace("pi", "π") + (" (= CP for worst input)" if ls == "--" else ""))
    a.set_yscale("log"), a.set_xlabel(r"Number of $\pi$ pulses"), a.set_ylabel(r"$1-C$ (worst input state)")
    a.text(0.03, 0.9, "1 % amplitude, 1 % detuning errors", transform=a.transAxes, fontsize=5.5, color="0.35")
    a.legend(fontsize=6, loc="center right")

    a = ax[1, 1]
    du = res["du"]
    xs = ["park 0 ms", "park 1.5 ms", "park 2.5 ms", "0.45 pi", "0.56 pi", "0.65 pi"]
    xl = ["π/2\n0 ms", "π/2\n1.5 ms", "π/2\n2.5 ms", "0.45π", "0.56π", "0.65π"]
    for (label, col, mk) in (("leg intensity drift", "#c2452d", "o"), ("leg ellipticity drift", "#1f5fa8", "s"),
                             ("white RIN only", "#a0a0a0", "^")):
        a.plot(range(6), [du[label][k] for k in xs], mk, ms=3, color=col, mfc="none", mew=0.7,
               label=f"model: {label}")
    for i in (2, 5):  # 'lost' / '< 0.1'
        a.plot([i, i], [0.0, 0.1], color="k", lw=2.0, solid_capstyle="butt")
    a.plot([3], [0.65], "D", ms=4, color="k")
    a.plot([], [], "D", ms=4, color="k", label="Du (2025), measured (bars: upper bounds)")
    a.set_xticks(range(6), xl, fontsize=6), a.set_ylim(-0.03, 1.05)
    a.set_ylabel(r"Echo contrast, $|3,\pm1\rangle$")
    a.legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 0.17))

    for axis, letter in zip(ax.flat, "abcd"):
        axis.text(-0.2, 1.02, letter, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
