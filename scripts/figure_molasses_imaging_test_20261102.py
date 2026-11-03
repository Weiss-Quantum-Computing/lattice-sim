"""Figure: X imaging at fixed dose and the full-depth dark lifetime, -40.5 MHz (2026-11-02/03, runs 005-011).

From data/measured/molasses_imaging_test_20261102_{imaging,lifetime}.csv
(scripts/molasses_imaging_test_20261102.py):

a  hops per atom per image cycle against imaging power (exposure chosen for ~15k counts), with the
   budget model's thermal hops added to the pooled 005-010 level;
b  losses per atom per image cycle against the molasses-on time per image set (9 planes x exposure);
c  separation (image 1, category 0) against power;
d  kept fraction against dark hold time with the repump on (006) and off (007), exponential fits
   (solid) and, for 006, the fit with a heating term (dashed).

Writes figures/molasses_imaging_test_20261102.png.
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
sys.path.insert(0, str(REPO / "scripts"))
from lattice_sim.viz.style import DOUBLE_COLUMN_IN, apply  # noqa: E402
import molasses_imaging_budget_20261102 as bud  # noqa: E402

MEAS = REPO / "data" / "measured"
COL = {"006": "#D55E00", "007": "#0072B2"}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def main():
    apply()
    im = read_csv(MEAS / "molasses_imaging_test_20261102_imaging.csv")
    lt = read_csv(MEAS / "molasses_imaging_test_20261102_lifetime.csv")
    f = lambda rows, k: np.array([float(r[k]) for r in rows])  # noqa: E731
    p, t = f(im, "power_uw"), f(im, "exposure_ms")
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN * 0.8, 4.4), constrained_layout=True)
    ax = ax.ravel()
    ax[0].errorbar(p, f(im, "hops_pct"), f(im, "hops_err_pct"), fmt="ko", ms=3, elinewidth=0.6, label="measured")
    base = sum(float(r["hops"]) for r in im[:-1]) / sum(float(r["atoms1"]) for r in im[:-1]) * 100
    pp = np.linspace(1.6, 3.2, 100)
    tt = bud.TARGET_COUNTS / (6190 * pp / (1 + pp / bud.PSAT_UW)) * 100
    ax[0].plot(pp, base + 100 * bud.thermal_hops_per_set(0.774 * pp, 9 * tt / 1000), "-", color="#CC79A7", lw=0.8,
               label="005–010 level + model thermal")
    for x, y, tm in zip(p, f(im, "hops_pct"), t):
        ax[0].annotate(f"{tm:.0f} ms", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=5)
    ax[0].set_xlabel("Imaging power (µW), ≈15k counts per plane")
    ax[0].set_ylabel("Hops per atom per cycle (%)")
    ax[0].legend(fontsize=5.5, loc="upper left")
    ax[1].errorbar(9 * t / 1000, f(im, "losses_pct"), f(im, "losses_err_pct"), fmt="ko", ms=3, elinewidth=0.6)
    ax[1].set_xlabel("Molasses-on time per image set (s)")
    ax[1].set_ylabel("Losses per atom per cycle (%)")
    ax[2].plot(p, f(im, "sep"), "ko", ms=3)
    ax[2].set_xlabel("Imaging power (µW)")
    ax[2].set_ylabel("Separation, image 1, no neighbours (σ)")
    ax[2].set_ylim(5, 6.5)
    for run in ("006", "007"):
        rr = [r for r in lt if r["run"] == run]
        x = f(rr, "hold_s")
        a, tau, et = float(rr[0]["fit_A"]), float(rr[0]["fit_tau_s"]), float(rr[0]["fit_tau_err_s"])
        ax[3].errorbar(x, f(rr, "kept_frac"), f(rr, "kept_err"), fmt="o", color=COL[run], ms=2.5, elinewidth=0.5,
                       label=f"{rr[0]['condition']}: τ = {tau:.0f} ± {et:.0f} s")
        xx = np.linspace(0, 60, 200)
        ax[3].plot(xx, a * np.exp(-xx / tau), "-", color=COL[run], lw=0.8)
        if run == "006":
            a2, tb, th = (float(rr[0][k]) for k in ("heat_A", "heat_tau_bg_s", "heat_t_h_s"))
            ax[3].plot(xx, a2 * np.exp(-xx / tb - (xx / th) ** 2), "--", color=COL[run], lw=0.8,
                       label=f"with heating term: t_h = {th:.0f} s")
    ax[3].set_xlabel("Dark hold at full depth (s)")
    ax[3].set_ylabel("Kept fraction")
    ax[3].legend(fontsize=5.5, loc="lower left")
    for a_, t_ in zip(ax, "abcd"):
        a_.text(-0.2, 1.03, t_, transform=a_.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_imaging_test_20261102.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
