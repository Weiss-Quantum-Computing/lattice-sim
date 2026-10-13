"""Figure: shape-robust suite ramps RS25 (2.5 ms) and RS175 (1.75 ms) against the
suite's S25 and N175, under the ER316 calibration and the first-principles
model with the measured extinction profile (h = -1, +1).

(a) Rise theta(t). (b, c) 1 - P0 at the end of the rise versus the
trap-frequency scale (relative to K0 = 0.94), worst over |3,-3> and |4,-3>.
(d) 1 - P0 after one full up-hold-down record versus the hold time, worst over
both states and scales x0.85-1.15.

Reads the target pairs from SUITE_DIR. Writes figures/suite_shape_robust.png
and data/suite_shape_robust_scan.json.
Usage: python scripts/figure_suite_shape_robust.py SUITE_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_suite_shape_robust import STATES, shape_splines  # noqa: E402
from score_suite import read_pair, rise_end_index  # noqa: E402
from lattice_sim.motion.fast_gaussian import propagate_scenarios  # noqa: E402
from lattice_sim.viz.style import COLORS, apply  # noqa: E402

DT = 0.5
SHAPE_STYLE = {"cal": ("-", "ER316 calibration"), "model_h-1": ("--", "model, h = -1"),
               "model_h+1": (":", "model, h = +1")}
RAMPS = {"S25": COLORS["red"], "RS25": COLORS["blue"], "N175": COLORS["ochre"], "RS175": COLORS["teal"]}


def rise_of(folder, stem):
    t, th, k0, _ = read_pair(folder, stem)
    i = rise_end_index(th)
    i0 = int(np.argmax(th > 1e-9)) - 1
    return t[i0: i + 1] - t[i0], th[i0: i + 1], k0


def p0_end(thetas_list, shapes, scales, k0):
    """1 - P0 at the end of each theta(t) on the common DT grid; shape (n_ramps, shapes, states, scales)."""
    x0, om = [], []
    for th in thetas_list:
        for shape in shapes:
            for st in STATES:
                cx, cf = shapes[shape][st]
                xx, ff = cx(th), cf(th)
                for sc in scales:
                    x0.append(xx)
                    om.append(2 * np.pi * ff * k0 * sc)
    n = len(thetas_list[0])
    p = propagate_scenarios(np.arange(n) * DT, np.array(x0), np.array(om), sample_every=n - 1)[:, -1]
    return (1 - p).reshape(len(thetas_list), len(shapes), len(STATES), len(scales))


def main(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply()
    shapes = shape_splines()
    scales = np.round(np.arange(0.75, 1.2501, 0.01), 2)
    out = {}
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for stem, col in RAMPS.items():
        t, th, k0 = rise_of(folder, stem)
        ax[0, 0].plot(t / 1e3, np.rad2deg(th), color=col, label=stem)
        tg = np.arange(0, t[-1] + 1e-9, DT)
        thg = np.interp(tg, t, th)
        inf = p0_end([thg], shapes, scales, k0)[0].max(axis=1)      # (shapes, scales), worst state
        panel = ax[0, 1] if stem.endswith("25") else ax[1, 0]
        for k, shape in enumerate(shapes):
            ls, _ = SHAPE_STYLE[shape]
            panel.semilogy(scales, np.maximum(inf[k], 1e-7), ls=ls, color=col)
        # full record vs hold: 200 us lead | rise | hold | fall | lead, padded to a common length
        holds = np.arange(0, 1000.1, 10.0)
        nr = tg.size
        lead = np.zeros(int(200 / DT))
        recs = []
        total = 2 * lead.size + 2 * nr + int(1000 / DT) + 1
        for hold in holds:
            r = np.r_[lead, thg, np.full(int(hold / DT), thg[-1]), thg[::-1], lead]
            recs.append(np.r_[r, np.zeros(total - r.size)])
        rob = scales[(scales >= 0.85 - 1e-9) & (scales <= 1.15 + 1e-9)]
        rob = rob[::3]
        end = p0_end(recs, shapes, rob, k0)                           # (holds, shapes, states, scales)
        worst = end.max(axis=(2, 3))
        for k, shape in enumerate(shapes):
            ls, _ = SHAPE_STYLE[shape]
            ax[1, 1].semilogy(holds, np.maximum(worst[:, k], 1e-7), ls=ls, color=col)
        rob_mask = (scales >= 0.85 - 1e-9) & (scales <= 1.15 + 1e-9)
        out[stem] = dict(rise_us=float(t[-1]),
                         rise_worst_x085_115={s: float(inf[k, rob_mask].max()) for k, s in enumerate(shapes)},
                         rise_worst_x075_125={s: float(inf[k].max()) for k, s in enumerate(shapes)},
                         record_end_worst_over_hold_0_1ms={s: float(worst[:, k].max()) for k, s in enumerate(shapes)},
                         record_end_hold_500us={s: float(worst[np.argmin(np.abs(holds - 500)), k]) for k, s in enumerate(shapes)})
        print(stem, json.dumps(out[stem]), flush=True)
    ax[0, 0].set(xlabel="Time (ms)", ylabel=r"Rotation $\theta$ (deg)")
    ax[0, 0].legend(loc="lower right")
    for a, title in ((ax[0, 1], "2.5 ms: S25, RS25"), (ax[1, 0], "1.75 ms: N175, RS175")):
        a.axvspan(0.85, 1.15, color="0.92", zorder=0)
        a.set(xlabel=r"Trap-frequency scale (relative to $K_0$ = 0.94)", ylabel=r"$1-P_0$ at end of rise", ylim=(1e-7, 0.3))
        a.set_title(title, fontsize=7)
    for shape, (ls, label) in SHAPE_STYLE.items():
        ax[0, 1].plot([], [], ls=ls, color="k", label=label)
    ax[0, 1].legend(loc="upper right")
    ax[1, 1].set(xlabel=r"Hold at 180$^\circ$ ($\mu$s)", ylabel=r"$1-P_0$ after one record", ylim=(1e-7, 0.3))
    ax[1, 1].set_title(r"worst over both states, scale $\times$0.85–1.15", fontsize=7)
    for a, letter in zip(ax.flat, "abcd"):
        a.text(-0.16, 1.02, letter, transform=a.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/suite_shape_robust.png")
    with open("data/suite_shape_robust_scan.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote figures/suite_shape_robust.png")


if __name__ == "__main__":
    import os
    folder = str(Path(sys.argv[1]).resolve())
    os.chdir(Path(__file__).resolve().parents[1])
    main(folder)
