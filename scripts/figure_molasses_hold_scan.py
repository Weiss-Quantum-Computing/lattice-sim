"""Figure: survival and hops against the molasses power during a 5 s hold in the X and X2 lattices (2026-10-30).

From data/measured/molasses_hold_scan_20261030.csv (runs 012, 013 X lattice; 016 X2) and
data/measured/molasses_hold_moves_20261030.csv:

a  survival over the hold (atoms of image 1 not lost) against the hold power;
b  hops per atom over the hold (log), with the Arrhenius fits of c;
c  ln(hops per atom - imaging background) against 1/P over the onset of hopping:
   for T = cP, h = Gamma_a tau exp(-U/kT) gives a straight line of slope -U/(kc);
d  nearest-neighbour hop-loss pairs above chance, by direction (X: 2.55-3.55 uW; X2: 1.05-3.55 uW).

Prints the fits: slope b (uW), kT/U at 2 and 3 uW, the attempt rate (exponential form and with the erfc
prefactor of lattice_sim.molasses.tables), and the hops extrapolated to 1.5 and 2.0 uW.
Writes figures/molasses_hold_scan.png.
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
HOLD_S = 5.0
IMAGING_HOPS = 0.003           # hops per atom from the two image sets alone (runs 009, 011: 0.2-0.3 %)
FIT_RANGE = {"X": (2.45, 3.65), "X2": (2.35, 3.25)}   # onset of hopping, below the saturation of the hop fraction
COL = {"X": "#0072B2", "X2": "#D55E00"}
DIRS = ("+x", "-x", "+y", "-y", "+z", "-z")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def scan(rows, runs):
    r = [x for x in rows if x["run"] in runs and x["hold_power_uw"] != ""]
    p = np.array([float(x["hold_power_uw"]) for x in r])
    n = np.array([float(x["atoms_img1"]) for x in r])
    h = np.array([float(x["hops"]) for x in r])
    lo = np.array([float(x["losses"]) for x in r])
    o = np.argsort(p)
    return p[o], n[o], h[o], lo[o]


def arrhenius(p, n, h, rng):
    s = (p > rng[0]) & (p < rng[1])
    y = np.clip(h[s] / n[s] - IMAGING_HOPS, 1e-3, None)
    b, a = np.polyfit(1 / p[s], np.log(y), 1, w=np.sqrt(h[s] + 1))
    return a, b, s


def main():
    apply()
    rows = read_csv(MEAS / "molasses_hold_scan_20261030.csv")
    data = {"X": scan(rows, ("013",)), "X2": scan(rows, ("016",))}
    x12 = scan(rows, ("012",))
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN * 0.75, 4.4), constrained_layout=True)
    ax = ax.ravel()
    for lat, (p, n, h, lo) in data.items():
        s = 1 - lo / n
        ax[0].errorbar(p, s, np.sqrt(np.clip(s * (1 - s), 1 / n, None) / n), color=COL[lat], marker="o", ms=2.5,
                       lw=0.7, capsize=0, label=f"{lat} lattice")
        f = h / n
        ax[1].errorbar(p, 100 * f, 100 * np.sqrt(np.maximum(h, 1)) / n, color=COL[lat], marker="o", ms=2.5, lw=0.7,
                       capsize=0)
        a, b, sel = arrhenius(p, n, h, FIT_RANGE[lat])
        pp = np.linspace(1.4, FIT_RANGE[lat][1], 100)
        ax[1].plot(pp, 100 * (np.exp(a + b / pp) + IMAGING_HOPS), color=COL[lat], lw=0.6, ls="--")
        ax[2].plot(1 / p[sel], np.log(np.clip(f[sel] - IMAGING_HOPS, 1e-3, None)), "o", color=COL[lat], ms=3)
        ii = np.linspace(1 / FIT_RANGE[lat][1], 1 / FIT_RANGE[lat][0], 20)
        ax[2].plot(ii, a + b * ii, color=COL[lat], lw=0.8, label=f"{lat}: slope {b:.1f} µW")
        x2 = -b / 2.0
        print(f"{lat}: b = {-b:.1f} uW, kT/U = {2 / -b:.3f} at 2 uW, {3 / -b:.3f} at 3 uW; Gamma_a = {np.exp(a) / HOLD_S:.0f} s^-1 "
              f"(exp form), {np.exp(a) / HOLD_S * np.sqrt(np.pi * x2):.0f} s^-1 (erfc form at 2 uW); hops per {HOLD_S:g} s: "
              f"{100 * np.exp(a + b / 2.0):.3f} % at 2.0 uW, {100 * np.exp(a + b / 1.5):.4f} % at 1.5 uW")
        lethal = p[(s < 0.5) & (p > 0) & (p < 2)]
        print(f"   low-power window with survival < 0.5: {lethal.min():.1f}-{lethal.max():.1f} uW; plateau (>= 0.85) "
              f"{p[(s >= 0.85) & (p > 0)].min():.1f}-{p[(s >= 0.85)].max():.1f} uW")
    s12 = 1 - x12[3] / x12[1]
    ax[0].plot(x12[0], s12, "o", color=COL["X"], mfc="none", ms=3, label="X, run 012")
    ax[0].set_xlabel("Molasses power during the hold (µW)")
    ax[0].set_ylabel(f"Survival over the {HOLD_S:g} s hold")
    ax[0].legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.42, 0.02))
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.1, 60)
    ax[1].set_xlabel("Molasses power during the hold (µW)")
    ax[1].set_ylabel(f"Hops per atom over the hold (%)")
    ax[2].set_xlabel("1 / P (µW$^{-1}$)")
    ax[2].set_ylabel("ln(hops per atom − imaging)")
    ax[2].legend(fontsize=5.5)

    mv = read_csv(MEAS / "molasses_hold_moves_20261030.csv")
    bands = {"X": [r for r in mv if r["run"] == "013" and r["power_band_uw"] == "2.55-3.55"],
             "X2": [r for r in mv if r["run"] == "016" and r["power_band_uw"] in ("1.05-2.55", "2.55-3.55")]}
    w = 0.38
    for k, (lat, rr) in enumerate(bands.items()):
        ex = [sum(float(r[f"{d}_real"]) - float(r[f"{d}_chance"]) for r in rr) for d in DIRS]
        ax[3].bar(np.arange(6) + (k - 0.5) * w, ex, w, color=COL[lat], label=lat)
        print(f"{lat} moves above chance:", dict(zip(DIRS, np.round(ex, 0))))
    ax[3].axhline(0, color="#000000", lw=0.6)
    ax[3].set_xticks(np.arange(6), DIRS)
    ax[3].set_xlabel("Move direction")
    ax[3].set_ylabel("Hop–loss pairs above chance")
    ax[3].legend(fontsize=5.5)
    for a, t in zip(ax, "abcd"):
        a.text(-0.2, 1.03, t, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_hold_scan.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    os.chdir(REPO)
    main()
