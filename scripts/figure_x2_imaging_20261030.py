"""Figure: imaging in the X2 lattice at fixed dose (2026-10-30, runs 023-026).

From data/measured/x2_imaging_20261030_{summary,displacement_hist,displacement_planes}.csv:

a  distribution of the floating-centre displacement of each atom in the state image (CameraGui
   state analysis, no state preparation), per setting, log scale;
b  fraction of atoms displaced by more than 1 px against plane;
c  per image cycle: hops and losses per atom (re-thresholded occupancy), and the fractions of atoms
   displaced beyond +2.28 px and below -1.41 px (the calibrated state-II and state-I displacements
   with the opposite sign), the motion that would cross the state boundary at 0.

Prints the summary table. Writes figures/x2_imaging_20261030.png.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lattice_sim.viz.style import DOUBLE_COLUMN_IN, apply  # noqa: E402

MEAS = REPO / "data" / "measured"
COL = {"023": "#D55E00", "024": "#E69F00", "025": "#009E73", "026": "#0072B2"}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def main():
    apply()
    summ = {r["run"]: r for r in read_csv(MEAS / "x2_imaging_20261030_summary.csv")}
    hist = read_csv(MEAS / "x2_imaging_20261030_displacement_hist.csv")
    planes = read_csv(MEAS / "x2_imaging_20261030_displacement_planes.csv")
    lab = {k: f"{float(r['exposure_ms']):.0f} ms × {float(r['power_uw']):g} µW" for k, r in summ.items()}
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    for run in summ:
        rr = [r for r in hist if r["run"] == run]
        x = np.array([float(r["bin_low_px"]) for r in rr]) + 0.125
        n = np.array([float(r["count"]) for r in rr])
        ax[0].step(x, n / n.sum(), where="mid", color=COL[run], lw=0.9)
        pp = [r for r in planes if r["run"] == run]
        ax[1].plot([int(r["plane"]) for r in pp], [100 * float(r["frac_absdisp_gt_1"]) for r in pp], "o-",
                   color=COL[run], ms=3, lw=0.8, label=lab[run])
    for xv in (-1.41, 2.28):
        ax[0].axvline(xv, color="#777777", ls=":", lw=0.7)
    ax[0].set_yscale("log")
    ax[0].set_xlim(-4.5, 4.5)
    ax[0].set_xlabel("Displacement in the state image (px)")
    ax[0].set_ylabel("Fraction of atoms per 0.25 px")
    ax[1].set_xlabel("Plane")
    ax[1].set_ylabel("Atoms displaced > 1 px (%)")
    ax[1].legend(fontsize=5.5, loc="upper left")
    runs = list(summ)
    keys = (("hops", "hops", None), ("losses", "losses", None), ("|d| > 1 px", "frac_absdisp_gt_1", 1),
            ("d > +2.28 px", "frac_disp_gt_2p28", 1), ("d < −1.41 px", "frac_disp_lt_m1p41", 1))
    w = 0.2
    for k, run in enumerate(runs):
        r = summ[run]
        vals = [100 * (float(r[c]) if f else float(r[c]) / float(r["atoms_img1"])) for _, c, f in keys]
        ax[2].bar(np.arange(len(keys)) + (k - 1.5) * w, vals, w, color=COL[run])
        print(run, lab[run], f"bright {r['brightness']} sep {r['separation_sigma']} miscall {r['miscall_per_site']}",
              " ".join(f"{n} {v:.2f}%" for (n, _, _), v in zip(keys, vals)))
    ax[2].set_xticks(np.arange(len(keys)), [n for n, _, _ in keys], rotation=30, ha="right")
    ax[2].set_ylabel("Per atom per image cycle (%)")
    for a, t in zip(ax, "abc"):
        a.text(-0.2, 1.03, t, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "x2_imaging_20261030.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
