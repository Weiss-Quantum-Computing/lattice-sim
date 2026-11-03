"""Figure: molasses detuning scan, 500 ms hold in a shallow X lattice (2026-11-02, runs 001-004).

From data/measured/molasses_detuning_20261102_{points,runs}.csv (scripts/molasses_detuning_20261102.py):

a  atoms kept on their site (1 - losses / atoms of image 1) and atom-number ratio (image 2 / image 1)
   against molasses power during the hold, per detuning; error bars binomial;
b  hops (empty -> filled sites) per atom of image 1 against power;
c  nearest-neighbour hop-loss pairs above chance by direction (loss site -> hop site), summed over
   0.3-3.2 uW, per 1000 atoms of image 1 in that range;
d  brightness of image 1 (1.75 uW x 150 ms at the hold detuning) against detuning, with the nominal
   Lorentzian normalised at -26.5 MHz and the best Lorentzian with a free lock offset (printed);
e  kept fraction against power x relative measured brightness (an empirical scattering-rate axis in
   uW at -26.5 MHz);
f  peak kept fraction (3-point mean) and the power window above 0.5 and 0.9 against detuning.

Writes figures/molasses_detuning_20261102.png.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lattice_sim.viz.style import DOUBLE_COLUMN_IN, apply  # noqa: E402

MEAS = REPO / "data" / "measured"
GAMMA_MHZ = 5.234
COL = {"001": "#D55E00", "002": "#E69F00", "003": "#009E73", "004": "#0072B2"}
DIRS = ["+x", "-x", "+y", "-y", "+z", "-z"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def lorentz(d, a, d0):
    return a / (1 + 4 * ((d - d0) / GAMMA_MHZ) ** 2)


def model_temperature_per_photon():
    """model T / scattering rate at the highest intensity of data/molasses/intensity.jsonl (full depth, 0 field)."""
    import json
    with open(REPO / "data" / "molasses" / "intensity.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    for d in (-25.5, -30.0, -35.5, -40.0):
        top = max((r for r in rows if r["detuning"] == d), key=lambda r: r["intensity"])
        print(f"model {d:+.1f} MHz: {np.mean(top['t_uk']) / top['scatter_rate'] * 1e6:.2f} uK per 1e6 s^-1 "
              f"at {top['intensity']} mW/cm^2")


def main():
    apply()
    pts = read_csv(MEAS / "molasses_detuning_20261102_points.csv")
    runs = {r["run"]: r for r in read_csv(MEAS / "molasses_detuning_20261102_runs.csv") if r["run"] in COL}
    b_ref = float(runs["001"]["bright"])
    fig, ax = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 4.6), constrained_layout=True)
    ax = ax.ravel()
    for run, r in runs.items():
        rr = [q for q in pts if q["run"] == run]
        p = np.array([float(q["power_uw"]) for q in rr])
        n1 = np.array([float(q["atoms1"]) for q in rr])
        keep = 1 - np.array([float(q["losses"]) for q in rr]) / np.maximum(n1, 1)
        ratio = np.array([float(q["atoms2"]) for q in rr]) / np.maximum(n1, 1)
        hops = np.array([float(q["hops"]) for q in rr]) / np.maximum(n1, 1)
        err = np.sqrt(np.clip(keep * (1 - keep), 1 / n1, None) / n1)
        lab = f"{float(r['detuning_mhz']):g} MHz"
        ax[0].errorbar(p, keep, err, fmt="o-", color=COL[run], ms=2, lw=0.7, elinewidth=0.5, label=lab)
        ax[0].plot(p, ratio, ":", color=COL[run], lw=0.7)
        ax[1].errorbar(p, 100 * hops, 100 * np.sqrt(np.clip(hops, 1 / n1, None) / n1), fmt="o-", color=COL[run],
                       ms=2, lw=0.7, elinewidth=0.5, label=lab)
        ax[4].plot(p * float(r["bright"]) / b_ref, keep, "o-", color=COL[run], ms=2, lw=0.7, label=lab)
        win = sum(float(q["atoms1"]) for q in rr if 0.3 <= float(q["power_uw"]) <= 3.2)
        ex = [1000 * (float(r[f"win_pairs_{d}"]) - float(r[f"win_chance_{d}"])) / win for d in DIRS]
        k = list(runs).index(run)
        ax[2].bar(np.arange(6) + (k - 1.5) * 0.2, ex, 0.2, color=COL[run], label=lab)
    ax[0].set_xlim(0, 3.5)
    ax[0].set_xlabel("Molasses power in hold (µW)")
    ax[0].set_ylabel("Fraction after 500 ms hold")
    ax[0].legend(fontsize=5.5, loc="upper right", title="solid: kept on site\ndotted: N₂/N₁", title_fontsize=5.5)
    ax[1].set_xlim(0, 3.5)
    ax[1].set_xlabel("Molasses power in hold (µW)")
    ax[1].set_ylabel("Hops per atom (%)")
    ax[2].axhline(0, color="#777777", lw=0.5)
    ax[2].set_xticks(np.arange(6), DIRS)
    ax[2].set_xlabel("Move direction (loss → hop site)")
    ax[2].set_ylabel("NN pairs above chance per 1000 atoms")
    ax[4].set_xlim(0, 3.5)
    ax[4].set_xlabel("Power × measured brightness ratio (µW at −26.5 MHz)")
    ax[4].set_ylabel("Kept on site")

    det = np.array([float(r["detuning_mhz"]) for r in runs.values()])
    br = np.array([float(r["bright"]) for r in runs.values()])
    ax[3].plot(det, br / 1e3, "ko", ms=3, label="measured")
    dd = np.linspace(-43, -24, 200)
    ax[3].plot(dd, b_ref / 1e3 * lorentz(dd, 1, 0) / lorentz(-26.5, 1, 0), "--", color="#777777", lw=0.8,
               label="Lorentzian, nominal Δ")
    (a, d0), cov = curve_fit(lorentz, det, br, p0=[2e7, 40])
    ax[3].plot(dd, lorentz(dd, a, d0) / 1e3, "-", color="#CC79A7", lw=0.8, label=f"free offset ({d0:+.0f} MHz)")
    ax[3].set_xlabel("Detuning from F′ = 5 (MHz)")
    ax[3].set_ylabel("Brightness (10³ counts per exposure)")
    ax[3].legend(fontsize=5.5, loc="upper left")
    print(f"brightness ratios {np.round(br / b_ref, 3)}; nominal Lorentzian {np.round(lorentz(det, 1, 0) / lorentz(-26.5, 1, 0), 3)}")
    print(f"free-offset Lorentzian: offset {d0:+.1f} +- {np.sqrt(cov[1, 1]):.1f} MHz, rms residual "
          f"{np.sqrt(np.mean((br - lorentz(det, a, d0)) ** 2)):.0f} counts")

    pk = [float(r["peak_stay_3pt"]) for r in runs.values()]
    ax[5].plot(det, pk, "ko-", ms=3, lw=0.8, label="peak kept (3-pt mean)")
    ax[5].set_xlabel("Detuning from F′ = 5 (MHz)")
    ax[5].set_ylabel("Peak kept on site")
    for d, r in zip(det, runs.values()):
        lo, hi = float(r["low_edge_uw"]), float(r["high_edge_uw"])
        print(f"{d:+.1f} MHz: peak {float(r['peak_stay_3pt']):.3f} at {r['peak_power_uw']} uW; kept > 0.5 for "
              f"{lo}-{hi} uW (x{hi / lo:.2f}); > 0.9 for {r['plateau90_lo_uw']}-{r['plateau90_hi_uw']} uW")
        ax[5].annotate(f"{lo:.2f}–{hi:.2f} µW", (d, float(r["peak_stay_3pt"])), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=5)
    ax[5].set_xlim(-43, -24)
    ax[5].set_ylim(0.6, 1.0)
    for a_, t_ in zip(ax, "abcdef"):
        a_.text(-0.2, 1.03, t_, transform=a_.transAxes, fontweight="bold", fontsize=9, va="bottom")
    model_temperature_per_photon()
    out = REPO / "figures" / "molasses_detuning_20261102.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
