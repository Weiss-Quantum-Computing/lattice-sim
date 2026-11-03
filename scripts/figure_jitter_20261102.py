"""Figure: lattice position relative to the camera, run 018 of 2026-11-02 (scripts/jitter_20261102.py).

From data/measured/jitter_20261102_{pictures,shots,allan,summary}.csv:

a  the lattice position in every picture (rows and columns, camera px) against the time in the shot, every shot as
   a thin line with its mean removed, the median over shots in black; the two image sets and the 50 ms gap;
b  the Allan deviation of the within-shot series against the averaging time, with the white-noise floor
   (the per-picture precision / sqrt(n));
c  the per-shot mean position against the time of day, rows and columns, with the linear trend;
d  the 1/e spot widths against the picture index (median over shots);
e  the spot amplitude against the picture index;
f  the per-site offset pattern (the median offset of each site over the run minus the run's median), as arrows.

Writes figures/jitter_20261102.png.
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
COL = {"row": "#D55E00", "col": "#0072B2"}
UM_PER_PX = 0.52


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def main():
    apply()
    pics = read_csv(MEAS / "jitter_20261102_pictures.csv")
    shots = read_csv(MEAS / "jitter_20261102_shots.csv")
    al = read_csv(MEAS / "jitter_20261102_allan.csv")
    summ = {r["axis"]: r for r in read_csv(MEAS / "jitter_20261102_summary.csv")}
    nshots = max(int(r["shot"]) for r in pics)
    npic = max(int(r["picture"]) for r in pics)
    T = np.array(sorted(set(float(r["t_in_shot_s"]) for r in pics)))
    off = {a: np.full((nshots, 2 * npic), np.nan) for a in ("row", "col")}
    wr = np.full((nshots, 2 * npic), np.nan); wc = wr.copy(); amp = wr.copy()
    for r in pics:
        k = (int(r["set"]) - 1) * npic + int(r["picture"]) - 1
        s = int(r["shot"]) - 1
        if r["row_off_px"]:
            off["row"][s, k] = float(r["row_off_px"]); off["col"][s, k] = float(r["col_off_px"])
            wr[s, k] = float(r["w_rows_px"]); wc[s, k] = float(r["w_cols_px"]); amp[s, k] = float(r["amp"])
    sitemap = read_csv(MEAS / "jitter_20261102_sitemap.csv")
    fig, ax = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 4.6), constrained_layout=True)
    ax = ax.ravel()
    for a, dy in (("row", 0.0), ("col", -0.4)):
        w = off[a] - np.nanmean(off[a], 1)[:, None]
        for s in range(nshots):
            ax[0].plot(T, w[s] + dy, "-", color=COL[a], lw=0.3, alpha=0.35)
        ax[0].plot(T, np.nanmedian(w, 0) + dy, "k-", lw=1.0)
        ax[0].text(T[0], dy + 0.17, f"{'rows' if a == 'row' else 'columns'} (offset {dy:+.1f})", color=COL[a], fontsize=6)
    ax[0].axvspan(T[npic - 1] + 0.15, T[npic], color="0.85", lw=0)
    ax[0].set_xlabel("Time in the shot (s)")
    ax[0].set_ylabel("Position minus the shot mean (px)")
    ax[0].set_ylim(-0.7, 0.3)
    ax[0].text(0.02, 0.03, "thin: every shot; black: median over shots; grey: the gap between the sets", transform=ax[0].transAxes, fontsize=5.5)

    for a in ("row", "col"):
        rr = [r for r in al if r["axis"] == a]
        tau = np.array([float(r["tau_s"]) for r in rr]); ad = np.array([float(r["allan_px"]) for r in rr])
        n = np.array([float(r["pictures"]) for r in rr])
        ax[1].loglog(tau, ad, "o-", color=COL[a], ms=3, lw=0.8, label="rows" if a == "row" else "columns")
        ax[1].loglog(tau, float(summ[a]["floor_px"]) / np.sqrt(n), ":", color=COL[a], lw=0.8)
    ax[1].set_xlabel("Averaging time (s)")
    ax[1].set_ylabel("Allan deviation within a shot (px)")
    ax[1].legend(fontsize=6, loc="upper right", title="dotted: per-picture precision / √n", title_fontsize=6)
    ax[1].set_ylim(0.008, 0.1)
    ax[1].set_xticks([0.17, 0.34, 0.85, 1.7, 3.4])
    ax[1].set_xticklabels(["0.17", "0.34", "0.85", "1.7", "3.4"])
    ax[1].minorticks_off()

    t0 = None
    tmin = []
    for r in shots:
        h, m, s = r["timestamp"][-8:].split(":")
        t = 3600 * int(h) + 60 * int(m) + int(s)
        t0 = t if t0 is None else t0
        tmin.append((t - t0) / 60.0)
    tmin = np.array(tmin)
    for a in ("row", "col"):
        y = np.array([float(r[f"{a}_mean_px"]) for r in shots])
        ax[2].plot(tmin, y, "o-", color=COL[a], ms=3, lw=0.6, label=f"{'rows' if a == 'row' else 'columns'}: sd {float(summ[a]['sd_shot_px']):.3f} px, "
                   f"trend {1e3 * UM_PER_PX * float(summ[a]['shot_drift_px_per_min']):+.0f} nm/min")
        sl, ic = np.polyfit(tmin, y, 1)
        ax[2].plot(tmin, sl * tmin + ic, "--", color=COL[a], lw=0.6)
    ax[2].set_xlabel("Time of day since the first shot (min)")
    ax[2].set_ylabel("Per-shot mean position (px)")
    ax[2].legend(fontsize=6, loc="lower left")

    k = np.arange(1, 2 * npic + 1)
    ax[3].plot(k, np.nanmedian(wr, 0), "s-", color=COL["row"], ms=2.5, lw=0.6, label="rows")
    ax[3].plot(k, np.nanmedian(wc, 0), "^-", color=COL["col"], ms=2.5, lw=0.6, label="columns")
    ax[3].set_xlabel("Picture (1–20 set 1, 21–40 set 2)")
    ax[3].set_ylabel("1/e spot width (px)")
    ax[3].legend(fontsize=6, loc="upper right")
    ax[4].plot(k, np.nanmedian(amp, 0) / 1e3, "ko-", ms=2.5, lw=0.6)
    ax[4].set_xlabel("Picture (1–20 set 1, 21–40 set 2)")
    ax[4].set_ylabel("Spot amplitude (10³ counts)")
    ii = np.array([int(r["i"]) for r in sitemap if r["row_offset_px"]]); jj = np.array([int(r["j"]) for r in sitemap if r["row_offset_px"]])
    dr = np.array([float(r["row_offset_px"]) for r in sitemap if r["row_offset_px"]]); dc = np.array([float(r["col_offset_px"]) for r in sitemap if r["row_offset_px"]])
    ax[5].quiver(jj, ii, dc, dr, angles="xy", scale_units="xy", scale=0.25, width=0.006, color="#009E73")
    ax[5].quiver([0.5], [8.6], [0.2], [0.0], angles="xy", scale_units="xy", scale=0.25, width=0.006, color="k")
    ax[5].text(1.6, 8.5, "0.2 px", fontsize=6, va="center")
    ax[5].set_xlim(-0.5, 8.5); ax[5].set_ylim(-0.5, 9.2)
    ax[5].set_aspect("equal")
    ax[5].set_xlabel("Column j")
    ax[5].set_ylabel("Row i")
    ax[5].set_title("per-site offset pattern (rows sd {0:.2f}, columns sd {1:.2f} px)".format(float(summ["row"]["sitemap_sd_px"]), float(summ["col"]["sitemap_sd_px"])), fontsize=6.5)
    for a_, t_ in zip(ax, "abcdef"):
        a_.text(-0.2, 1.03, t_, transform=a_.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "jitter_20261102.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
