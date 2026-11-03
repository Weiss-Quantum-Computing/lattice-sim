"""Figure: X and X2 imaging at -40.5 MHz, 150 ms, no state preparation (2026-11-03, runs 013-017).

From data/measured/molasses_x2_imaging_20261102_{hist,shots,periodicity,summary}.csv
(scripts/molasses_x2_imaging_20261102.py):

a  fit-height histograms, no-neighbour category: 013 image 1 (X), 014 image 1 and 2 (X2 / X2),
   015 image 2 (X2 after an X image), each normalised to its number of sites;
b  floating-centre displacement of the state image (image 2) from the image-1 floating centre,
   014 (X2 -> X2) and 015-017 (X -> X2 with the X2 image at 2.2, 1.8, 1.7 uW), with the calibrated state centres (dashed) and boundary (dotted);
c  014: per-shot state-I fraction (boundary 0 px) against the per-shot median displacement;
d  015: coherence of the displacements with an X2 lattice of period p, against the X2 phase step per
   X column (a_X / a_X2 mod 1), at the best p; the 11.47 ratio of a simple standing wave is marked.

Writes figures/molasses_x2_imaging_20261102.png.
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
STATE_I, STATE_II = 1.41, -2.28


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def main():
    apply()
    hist = read_csv(MEAS / "molasses_x2_imaging_20261102_hist.csv")
    shots = read_csv(MEAS / "molasses_x2_imaging_20261102_shots.csv")
    per = read_csv(MEAS / "molasses_x2_imaging_20261102_periodicity.csv")
    summ = {r["run"]: r for r in read_csv(MEAS / "molasses_x2_imaging_20261102_summary.csv")}
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN * 0.8, 4.4), constrained_layout=True)
    ax = ax.ravel()

    def h(run, kind):
        rr = [r for r in hist if r["run"] == run and r["kind"] == kind]
        x = np.array([float(r["bin_lo"]) for r in rr])
        c = np.array([float(r["count"]) for r in rr])
        return x, c / c.sum()

    for run, kind, col, lab in (("013", "height_image1", "#000000", "013 image 1, X"),
                                ("014", "height_image1", "#D55E00", "014 image 1, X2"),
                                ("014", "height_image2", "#E69F00", "014 image 2, X2"),
                                ("015", "height_image2", "#0072B2", "015 image 2, X2 after X")):
        x, c = h(run, kind)
        ax[0].step(x / 1e3, c, where="post", color=col, lw=0.8, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_xlim(-5, 38)
    ax[0].set_xlabel("Fit height (10³ counts), no neighbours")
    ax[0].set_ylabel("Fraction of sites per 500 counts")
    ax[0].legend(fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 1.0), framealpha=1)
    ax[0].set_ylim(1e-5, 3)

    for run, col, lab in (("014", "#D55E00", "014, X2 → X2, 2.2 µW"), ("015", "#0072B2", "015, X → X2, 2.2 µW"),
                          ("016", "#009E73", "016, X → X2, 1.8 µW"), ("017", "#CC79A7", "017, X → X2, 1.7 µW")):
        x, c = h(run, "displacement")
        ax[1].step(x, c, where="post", color=col, lw=0.8,
                   label=f"{lab}: σ = {float(summ[run]['d_sigma']):.2f} px")
    for v, ls in ((STATE_I, "--"), (STATE_II, "--"), (0, ":")):
        ax[1].axvline(v, color="#777777", lw=0.6, ls=ls)
    ax[1].set_yscale("log")
    ax[1].set_xlim(-4, 4)
    ax[1].set_ylim(1e-4, 2)
    ax[1].set_xlabel("Displacement from image-1 centre (px)")
    ax[1].set_ylabel("Fraction of atoms per 0.1 px")
    ax[1].legend(fontsize=5.5, loc="upper left")

    s14 = [r for r in shots if r["run"] == "014"]
    ax[2].plot([float(r["offset_px"]) for r in s14], [float(r["state1_frac"]) for r in s14], "o", color="#D55E00", ms=3)
    ax[2].set_xlabel("Per-shot median displacement, 014 (px)")
    ax[2].set_ylabel("State-I fraction per shot")
    ax[2].text(0.04, 0.92, f"r = {float(summ['014']['corr_state1_frac_vs_offset']):.2f}", transform=ax[2].transAxes,
               fontsize=6)

    rf = [r for r in per if r["run"] == "015" and r["scan"] == "f"]
    f = np.array([float(r["value"]) for r in rf])
    ax[3].plot(f, [float(r["coherence"]) for r in rf], "-", color="#0072B2", lw=0.8)
    ax[3].axhline(float(summ["015"]["x2_null_max"]), color="#777777", lw=0.6, ls=":", label="shuffled null (max)")
    for v in (0.47, 0.53):
        ax[3].axvline(v, color="#CC79A7", lw=0.6, ls="--")
    ax[3].text(0.5, 0.35, "a_X / a_X2 = 11.47", color="#CC79A7", fontsize=5.5, ha="center",
               transform=ax[3].get_xaxis_transform())
    ax[3].set_xlabel("X2 phase step per X column (cycles)")
    ax[3].set_ylabel(f"Coherence at p = {float(summ['015']['x2_period_px']):.2f} px")
    ax[3].legend(fontsize=5.5, loc="upper left")
    for a_, t_ in zip(ax, "abcd"):
        a_.text(-0.2, 1.03, t_, transform=a_.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_x2_imaging_20261102.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
