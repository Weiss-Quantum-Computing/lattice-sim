"""Figure: the measured imaging map of 2026-10-26 (X lattice, 100 ms exposures).

From data/measured/molasses_imaging_20261026_reanalysis.csv (the re-thresholded
analysis in Camera Control/20261026/molasses_reanalysis):

a  atom brightness per 100 ms exposure against the delivered fiber power, from
   the +100 MHz AOM curve of 2026-10-29 (lattice_sim.molasses.aom; the +95 and
   +105 MHz points use the same curve, so their power axis is uncertain);
b  hops per atom between the two image sets against brightness: the trade
   that sets the imaging point, independent of the power calibration;
c  losses per atom against brightness.

The dotted lines mark the imaging targets of the re-analysis: 14 500 counts
per atom at 100 ms for readout fidelity > 0.9995 in every neighbour category,
and hops < 0.2 %. Writes figures/molasses_lab_map_20261026.png.
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
from lattice_sim.molasses import aom  # noqa: E402
from lattice_sim.viz.style import DOUBLE_COLUMN_IN, apply  # noqa: E402

DATA = REPO / "data" / "measured" / "molasses_imaging_20261026_reanalysis.csv"
FIDELITY_COUNTS = 14500        # per atom per 100 ms exposure (re-analysis target_check)
HOP_TARGET = 0.2               # % per atom per image cycle
COL = {30.5: "#009E73", 25.5: "#0072B2", 20.5: "#D55E00"}
AOM_MHZ = {30.5: 95, 25.5: 100, 20.5: 105}


def load(path=DATA):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def main():
    apply()
    rows = load()
    curves = aom.load_curves()
    p24 = float(aom.delivered(aom.IMAGING_REQUEST, aom.RESCALE_NOW, curves)[1])
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    for d, c in COL.items():
        m = sorted([r for r in rows if r["detuning_MHz"] == d], key=lambda r: -r["rescale"])
        b = np.array([r["brightness"] for r in m])
        n = np.array([r["atoms_img1"] for r in m])
        h = np.array([r["hops_per_atom_%"] for r in m])
        lo = np.array([r["losses_per_atom_%"] for r in m])
        p = p24 * np.array([float(aom.relative_power(r["rescale"], curves=curves)) for r in m])
        lab = f"−{d:g} MHz (+{AOM_MHZ[d]} MHz AOM)"
        ax[0].plot(p, b * 1e-3, color=c, marker="o", ms=3, lw=0.9, label=lab)
        ax[1].errorbar(b * 1e-3, h, 100 * np.sqrt(np.maximum(h / 100, 1 / n) / n), color=c, marker="o", ms=3,
                       lw=0.9, capsize=0)
        ax[2].errorbar(b * 1e-3, lo, 100 * np.sqrt(lo / 100 * (1 - lo / 100) / n), color=c, marker="o", ms=3,
                       lw=0.9, capsize=0)
        for a in ax[1:]:
            for r, x, y in zip(m, b, h if a is ax[1] else lo):
                if r["run"] in (3.0, 18.0, 12.0):
                    a.annotate(f"r{r['rescale']:g}", (x * 1e-3, y), fontsize=5, color=c, xytext=(2, 2),
                               textcoords="offset points")
    ax[0].set_xlabel("Delivered power, fiber total (µW)")
    ax[0].set_ylabel("Atom brightness (10$^3$ counts / 100 ms)")
    ax[0].legend(fontsize=5.5)
    ax[0].text(0.97, 0.03, "power from the +100 MHz\nAOM curve for all three", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=5.5, color="#555555")
    for a in ax[1:]:
        a.axvline(FIDELITY_COUNTS * 1e-3, color="#777777", ls=":", lw=0.8)
        a.set_xlabel("Atom brightness (10$^3$ counts / 100 ms)")
        a.set_yscale("log")
    ax[1].axhline(HOP_TARGET, color="#777777", ls=":", lw=0.8)
    ax[1].set_ylabel("Hops per atom per image cycle (%)")
    ax[2].set_ylabel("Losses per atom per image cycle (%)")
    ax[1].text(0.03, 0.97, "fidelity needs ≥ 14.5k (dotted)\nhop target 0.2 %", transform=ax[1].transAxes,
               va="top", fontsize=5.5, color="#555555")
    for a, s in zip(ax, "abc"):
        a.text(-0.2, 1.03, s, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_lab_map_20261026.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
