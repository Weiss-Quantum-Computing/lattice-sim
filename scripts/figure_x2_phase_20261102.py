"""Figure: the X2 lattice phase in the X -> X2 displacements of run 015 (2026-11-03).

The X2 period p, phase step per X column f and per row f_y come from
data/measured/molasses_x2_imaging_20261102_summary.csv (scripts/molasses_x2_imaging_20261102.py).
Each atom's X2 phase is estimated leave-one-out from the other atoms of its (shot, plane), so an atom
never sets its own reference:

a  one shot, three planes: displacement against column, with the X2 site positions of that plane
   (phase from all its atoms) as lines;
b  all atoms: raw displacement, and the displacement minus its X2 offset predicted from the other atoms
   of its (shot, plane) with the fitted column step; peaks at multiples of p appear only after the
   correction;
c  the same correction with the column step 0.47 of a_X / a_X2 = 11.47, and with the phase taken from
   a different shot (null);
d  the per-plane X2 offset within each shot relative to plane 1 (unwrapped): the phase moves between
   planes during an image set.

Writes figures/x2_phase_20261102.png.
Usage: python scripts/figure_x2_phase_20261102.py ["~/Desktop/Data/Camera Control/20261102"]
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
import camera_runs as cr  # noqa: E402

DATA = "~/Desktop/Data/Camera Control/20261102"
RUN = "015"
DMAX_PX = 2.5
EXAMPLE_SHOT, EXAMPLE_PLANES = None, (2, 4, 6)     # None: the shot with the most atoms
COL = ("#D55E00", "#009E73", "#0072B2")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def residuals(v, x, y, grp, p, f, fy, ref_grp=None):
    """d minus the X2 offset from the other atoms of the reference group (leave-one-out)."""
    z = np.exp(2j * np.pi * (v / p - f * x - fy * y))
    s = np.bincount(grp, z.real, minlength=grp.max() + 1) + 1j * np.bincount(grp, z.imag, minlength=grp.max() + 1)
    ref = s[grp] - z if ref_grp is None else s[ref_grp]
    phase = np.angle(ref) / (2 * np.pi)
    r = v - p * (phase + f * x + fy * y)
    return r - p * np.round((r - v) / p)          # keep the integer site number of the raw displacement


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DATA
    os.chdir(REPO)
    apply()
    summ = next(r for r in read_csv(REPO / "data" / "measured" / "molasses_x2_imaging_20261102_summary.csv")
                if r["run"] == RUN)
    p, f, fy = float(summ["x2_period_px"]), float(summ["x2_column_step"]), float(summ["x2_row_step"])
    f -= round(f)                                 # 0.94 cycles per column is -0.06 (same at integer columns)
    a, d = cr.state(folder, RUN)
    sel = (a > 0) & (np.abs(d) < DMAX_PX)
    sel[:, :1] = sel[:, 8:] = False
    s, z, y, x = np.nonzero(sel)
    v = d[sel]
    grp = s * 10 + z
    keep = np.bincount(grp)[grp] >= 5
    s, z, y, x, v, grp = s[keep], z[keep], y[keep], x[keep], v[keep], grp[keep]

    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN * 0.8, 4.6), constrained_layout=True)
    ax = ax.ravel()

    shot = EXAMPLE_SHOT if EXAMPLE_SHOT is not None else int(np.bincount(s).argmax())
    for k, (pl, c) in enumerate(zip(EXAMPLE_PLANES, COL)):
        m = (s == shot) & (z == pl)
        zc = np.exp(2j * np.pi * (v[m] / p - f * x[m] - fy * y[m])).mean()
        phi = np.angle(zc) / (2 * np.pi)
        xs = x[m] + (k - 1) * 0.18
        ax[0].plot(xs, v[m], "o", color=c, ms=2.5, label=f"plane {pl}")
        xx = np.linspace(-0.5, 8.5, 50)
        for n in range(-3, 4):
            ax[0].plot(xx, p * (n + phi + f * xx + fy * 4), "-", color=c, lw=0.4, alpha=0.6)
    ax[0].set_ylim(-2.2, 2.2)
    ax[0].set_xlim(-0.5, 8.5)
    ax[0].set_xlabel("X column")
    ax[0].set_ylabel(f"Displacement X → X2, shot {shot + 1} (px)")
    ax[0].legend(fontsize=5.5, loc="upper right", ncol=3)

    bins = np.arange(-2.5, 2.51, 0.05)
    r_fit = residuals(v, x, y, grp, p, f, fy)
    ax[1].hist(v, bins, histtype="step", color="#777777", lw=0.8, label="raw")
    ax[1].hist(r_fit, bins, histtype="step", color="#0072B2", lw=0.9,
               label=f"minus X2 offset ({f:+.2f} cycles/column)")
    for n in range(-2, 3):
        ax[1].axvline(n * p, color="#CC79A7", lw=0.5, ls=":")
    ax[1].set_xlabel("Displacement (px)")
    ax[1].set_ylabel("Atoms per 0.05 px")
    ax[1].legend(fontsize=5.5, loc="upper left")
    ax[1].text(0.98, 0.95, f"dotted: n × {p:.2f} px", transform=ax[1].transAxes, ha="right", va="top", fontsize=5.5,
               color="#CC79A7")

    r_bad = residuals(v, x, y, grp, p, 0.47, fy)
    shots = np.unique(s)
    other = {g: (shots[(np.searchsorted(shots, g // 10) + 7) % len(shots)]) * 10 + g % 10 for g in np.unique(grp)}
    ogrp = np.array([other[g] for g in grp])
    ok = np.isin(ogrp, grp)
    r_null = residuals(v[ok], x[ok], y[ok], grp[ok], p, f, fy, ref_grp=ogrp[ok])
    ax[2].hist(r_fit, bins, histtype="step", color="#0072B2", lw=0.9, label=f"step {f:+.2f} (fit)")
    ax[2].hist(r_bad, bins, histtype="step", color="#D55E00", lw=0.8, label="step 0.47 (a_X/a_X2 = 11.47)")
    ax[2].hist(r_null, bins, histtype="step", color="#777777", lw=0.8, label="phase from another shot")
    for n in range(-2, 3):
        ax[2].axvline(n * p, color="#CC79A7", lw=0.5, ls=":")
    ax[2].set_xlabel("Displacement minus predicted X2 offset (px)")
    ax[2].set_ylabel("Atoms per 0.05 px")
    ax[2].legend(fontsize=5.5, loc="upper left")

    phases = np.full((len(shots), 9), np.nan)
    for i, sh in enumerate(shots):
        for pl in range(1, 8):
            m = (s == sh) & (z == pl)
            if m.sum() >= 5:
                zc = np.exp(2j * np.pi * (v[m] / p - f * x[m] - fy * y[m])).mean()
                phases[i, pl] = p * np.angle(zc) / (2 * np.pi)
    rel = []
    for i in range(len(shots)):
        if np.isnan(phases[i, 1:8]).any():
            continue
        u = np.unwrap(phases[i, 1:8] * 2 * np.pi / p) * p / (2 * np.pi)
        rel.append(u - u[0])
        ax[3].plot(np.arange(1, 8), u - u[0], "-", color="#0072B2", lw=0.4, alpha=0.5)
    rel = np.array(rel)
    ax[3].plot(np.arange(1, 8), np.median(rel, 0), "k-", lw=1.2, label="median over shots")
    ax[3].axhline(0, color="#777777", lw=0.5)
    ax[3].set_xlabel("Plane (imaging order)")
    ax[3].set_ylabel("X2 offset change since plane 1 (px)")
    ax[3].legend(fontsize=5.5, loc="upper left")
    print(f"  plane 1 -> 7 offset change: median {np.median(rel[:, -1]):.3f} px, rms {np.sqrt(np.mean(rel[:, -1] ** 2)):.3f} px, "
          f"{len(rel)} shots; spread of plane-1 offsets {np.nanstd(phases[:, 1]):.3f} px")

    for a_, t_ in zip(ax, "abcd"):
        a_.text(-0.2, 1.03, t_, transform=a_.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "x2_phase_20261102.png"
    fig.savefig(out)
    print(f"p {p} px, f {f}, f_y {fy}; example shot {shot + 1}; wrote", out.relative_to(REPO))
    for lab, r in (("fit", r_fit), ("11.47", r_bad), ("other shot", r_null)):
        fr = np.mean(np.abs(r - p * np.round(r / p)) < p / 6)
        print(f"  {lab}: fraction within p/6 of a site {fr:.3f} (uniform 0.333)")


if __name__ == "__main__":
    main()
