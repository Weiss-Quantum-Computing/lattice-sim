"""Figure: the 2026-10-30 PGC-field scans and the loss against position in the block.

From data/measured/molasses_pgc_field_20261030.csv and data/measured/molasses_spatial_profiles_20261030.csv:

a  survival over the 5 s hold (X lattice, full depth, molasses 0.5 uW) against the field on one axis, the
   others at 0; both runs per axis (filled: first, open: repeat);
b  loss against plane (z index) relative to the mean loss of each regime: low power (PGC scans at
   |B| <= 40 mG; X 0.4-0.6 uW; X2 1.0-1.3 uW) against high power (X 2.7-3.6 uW; X2 2.5-3.3 uW);
c  the same against row (y);
d  the same against column (x).

Prints, per axis and per run, the plateau survival and the half-maximum crossings (centre and full
width where both sides cross). Writes figures/molasses_pgc_field_20261030.png.
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
AX_COL = {"x": "#0072B2", "y": "#009E73", "z": "#D55E00"}
REG = {  # regime: (label, colour, line style)
    "pgc_0.5uW_|B|<=40mG": ("X, 0.5 µW (PGC scans)", "#0072B2", "-"),
    "X_0.4-0.6uW": ("X, 0.4–0.6 µW", "#56B4E9", "-"),
    "X2_1.0-1.3uW": ("X2, 1.0–1.3 µW", "#009E73", "-"),
    "X_2.7-3.6uW": ("X, 2.7–3.6 µW", "#D55E00", "--"),
    "X2_2.5-3.3uW": ("X2, 2.5–3.3 µW", "#CC79A7", "--"),
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def survival(rows):
    b = np.array(sorted({float(r["field_mG"]) for r in rows}))
    n = np.array([sum(float(r["atoms_img1"]) for r in rows if float(r["field_mG"]) == x) for x in b])
    lo = np.array([sum(float(r["losses"]) for r in rows if float(r["field_mG"]) == x) for x in b])
    s = 1 - lo / n
    return b, s, np.sqrt(np.clip(s * (1 - s), 1 / n, None) / n)


def half_max(b, s):
    sm = np.convolve(s, np.ones(3) / 3, "same")
    j = int(np.argmax(sm[1:-1])) + 1
    top = sm[j]
    lo = hi = None
    for i in range(j, 0, -1):
        if s[i - 1] < top / 2 <= s[i]:
            lo = b[i - 1] + (top / 2 - s[i - 1]) / (s[i] - s[i - 1]) * (b[i] - b[i - 1])
            break
    for i in range(j, len(b) - 1):
        if s[i + 1] < top / 2 <= s[i]:
            hi = b[i] + (s[i] - top / 2) / (s[i] - s[i + 1]) * (b[i + 1] - b[i])
            break
    return top, lo, hi


def main():
    apply()
    scans = read_csv(MEAS / "molasses_pgc_field_20261030.csv")
    prof = read_csv(MEAS / "molasses_spatial_profiles_20261030.csv")
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN * 0.75, 4.4), constrained_layout=True)
    ax = ax.ravel()
    for axis in "xyz":
        runs = sorted({r["run"] for r in scans if r["axis"] == axis})
        for k, run in enumerate(runs):
            b, s, e = survival([r for r in scans if r["run"] == run])
            ax[0].errorbar(b, s, e, color=AX_COL[axis], marker="o", ms=2.5, lw=0.6, capsize=0,
                           mfc=AX_COL[axis] if k == 0 else "none", label=f"{axis}" if k == 0 else None)
            top, lo, hi = half_max(b, s)
            txt = f"{run} {axis}: plateau {top:.3f}, s(0) {s[b == 0][0]:.3f}, half max at {lo} / {hi}"
            if lo is not None and hi is not None:
                txt += f" -> centre {(lo + hi) / 2:.1f} mG, FWHM {hi - lo:.0f} mG"
            print(txt)
        b, s, _ = survival([r for r in scans if r["axis"] == axis])
        top, lo, hi = half_max(b, s)
        print(f"{axis} pooled: plateau {top:.3f}, half max at {lo} / {hi}")
    ax[0].set_xlabel("Field on the scanned axis (mG)")
    ax[0].set_ylabel("Survival over the 5 s hold")
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=5.5, loc="lower center", title="axis", title_fontsize=5.5)
    for a, axis, name in ((ax[1], "z", "Plane (z index)"), (ax[2], "y", "Row (y index)"), (ax[3], "x", "Column (x index)")):
        for reg, (lab, c, ls) in REG.items():
            rr = [r for r in prof if r["regime"] == reg and r["axis"] == axis]
            n = np.array([float(r["atoms_img1"]) for r in rr])
            lo = np.array([float(r["losses"]) for r in rr])
            f = lo / n
            m = lo.sum() / n.sum()
            a.errorbar(np.arange(len(f)), f / m, np.sqrt(f * (1 - f) / n) / m, color=c, ls=ls, marker="o", ms=2.5,
                       lw=0.8, capsize=0, label=lab)
            print(f"{reg} {axis}: loss {np.round(f, 2).tolist()} (mean {m:.3f})")
        a.axhline(1, color="#000000", lw=0.5)
        a.set_xlabel(name)
        a.set_ylabel("Loss / mean loss")
    ax[1].legend(fontsize=5, loc="upper center", ncol=2)
    for a, t in zip(ax, "abcd"):
        a.text(-0.2, 1.03, t, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_pgc_field_20261030.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
