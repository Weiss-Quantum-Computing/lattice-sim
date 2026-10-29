"""The molasses AOM calibration: Supertime's file against the measured curves.

Writes figures/molasses_aom_calibration.png and a Supertime calibration file
from the 2026-10-29 measurement, data/measured/OM-2026-10-29.txt (volts, uW,
tab separated, CRLF, as OM-2014-9-3.txt), with the dark reading at 0-1.1 V
(DARK_UW) subtracted and the top at VMAX = 4.4 V (20.91 uW), below RF saturation. Use --offset 0 to keep the readings as measured.

a  power against control voltage: the file as Supertime uses it (rescale 24),
   the earlier measurement and the 2026-10-29 one, with the voltages for the
   imaging (5) and PGC-hold (0.6) requests;
b  delivered / requested power against the request at rescale 24;
c  delivered power against the rescale value, relative to rescale 24, for the
   imaging request, with the 24/r the file promises.

Usage: python scripts/molasses_aom_calibration.py [--offset UW]
"""

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

DARK_UW = aom.BACKGROUND_UW["new"]     # meter reading with the beam blocked (room background)
COL = {"supertime": "#000000", "old": "#0072B2", "new": "#D55E00"}
LABEL = {"supertime": "Supertime file (rescale 24)", "old": "measured earlier",
         "new": "measured 2026-10-29 (background subtracted)"}


VMAX = 4.4               # top of the file: the lab keeps the AOM RF below saturation (Maarten, 2026-10-29: 4.4 V, 20.91 uW)


def write_supertime_file(curves, offset, path, vmax=VMAX):
    v, pw = curves["new"]
    keep = v <= vmax + 1e-9
    lines = aom.supertime_file_lines((v[keep], pw[keep]), offset_uw=offset)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")


def main(argv):
    offset = DARK_UW
    if "--offset" in argv:
        offset = float(argv[argv.index("--offset") + 1])
    apply()
    c = aom.load_curves()
    write_supertime_file(c, offset, REPO / "data" / "measured" / "OM-2026-10-29.txt")
    print(f"wrote data/measured/OM-2026-10-29.txt (offset {offset} uW)")

    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    v_file, p_file = c["supertime"]
    scaled = (v_file, p_file * aom.RESCALE_NOW / p_file.max())
    for k, cur in (("supertime", scaled), ("old", c["old"]), ("new", c["new"])):
        ax[0].plot(cur[0], cur[1], color=COL[k], marker="o", ms=2, lw=0.9, label=LABEL[k])
    for req, name, ha in ((aom.IMAGING_REQUEST, "imaging", "left"), (aom.PGC_HOLD_REQUEST, "PGC hold", "right")):
        v = float(aom.request_to_volts(req, aom.RESCALE_NOW, c["supertime"]))
        ax[0].axvline(v, color="#777777", lw=0.6, ls=":")
        pad = " " if ha == "left" else ""
        ax[0].text(v, 22.5, f"{pad}{name}{'' if pad else ' '}\n{pad}{v:.2f} V{'' if pad else ' '}", fontsize=5.5,
                   color="#555555", va="top", ha=ha)
    ax[0].set_xlabel("Control voltage (V)")
    ax[0].set_ylabel("Power out of the fiber (µW)")
    ax[0].legend(fontsize=5.5, loc="center right")

    req = np.geomspace(0.3, 20, 80)
    for k in ("old", "new"):
        _, p, _ = aom.delivered(req, aom.RESCALE_NOW, c, k)
        ax[1].plot(req, p / req, color=COL[k], lw=1, label=LABEL[k])
    for r in (aom.PGC_HOLD_REQUEST, aom.IMAGING_REQUEST):
        ax[1].axvline(r, color="#777777", lw=0.6, ls=":")
    ax[1].axhline(1, color="#000000", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Requested power (Supertime units)")
    ax[1].set_ylabel("Delivered / requested")
    ax[1].set_ylim(0, 1.2)
    ax[1].legend(fontsize=5.5)

    r = np.arange(12, 34, 1.0)
    ax[2].plot(r, 24 / r, color=COL["supertime"], lw=1, label="24/r (the file)")
    for k in ("old", "new"):
        ax[2].plot(r, aom.relative_power(r, curves=c, measured=k), color=COL[k], lw=1, label=LABEL[k])
    ax[2].set_xlabel("Rescale value r")
    ax[2].set_ylabel("Imaging power relative to r = 24")
    ax[2].legend(fontsize=5.5)
    for a, s in zip(ax, "abc"):
        a.text(-0.2, 1.03, s, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    fig.savefig(REPO / "figures" / "molasses_aom_calibration.png")
    print("wrote figures/molasses_aom_calibration.png")


if __name__ == "__main__":
    os.chdir(REPO)
    main(sys.argv)
