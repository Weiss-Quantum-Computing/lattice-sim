"""Figures: the imaging / PGC molasses in the full-depth lattice, from the
Monte Carlo tables of scripts/molasses_scan.py (lattice_sim.molasses).

figures/molasses_intensity.png (zero field)
  a  kinetic temperature (mean of x, y, z) against intensity per beam, one
     curve per detuning;
  b  photon scattering rate against intensity;
  c  hopping rate (sum over axes, Arrhenius with attempt rate 1/tau_E and the
     tail temperature of the ensemble, tables.Tables.hop_rate);
  d  scattering rate against temperature, parametric in intensity from the
     temperature minimum upward (below it the same T comes with fewer photons):
     the photon yield available at a given temperature.

figures/molasses_field.png
  a, b  temperature against a residual field along x, y, z at -51 MHz and
        at the closest detuning scanned (-10 MHz), three intensities;
  c     field at which the temperature doubles (mean over axes) against
        intensity, one curve per detuning (open: > the largest field scanned).

Usage: python scripts/figure_molasses.py
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
from lattice_sim.molasses import tables as T  # noqa: E402
from lattice_sim.molasses.model import arrhenius_hop_rate  # noqa: E402
from lattice_sim.viz.style import DOUBLE_COLUMN_IN, apply  # noqa: E402

AXIS_COLOURS = ("#0072B2", "#D55E00", "#009E73")
AXIS_STYLES = ("-", "--", ":")


def det_colours(n):
    return [matplotlib.colormaps["viridis"](v) for v in np.linspace(0.05, 0.85, n)][::-1]


def letter(ax, s, dx=-0.2):
    ax.text(dx, 1.03, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def figure_intensity(tab, out):
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 5.0), constrained_layout=True)
    cols = det_colours(len(tab.detunings))
    hop = tab.node_hop_rates().sum(axis=-1)
    for k, d in enumerate(tab.detunings):
        ok = np.isfinite(tab.scatter[k])
        ii = tab.intensities[ok]
        tm = tab.t[k][ok].mean(axis=1)
        te = np.sqrt((tab.t_err[k][ok] ** 2).sum(axis=1)) / 3
        kw = dict(color=cols[k], marker="o", ms=2.5, lw=0.9)
        ax[0, 0].errorbar(ii, tm, te, label=f"{d:+g} MHz", capsize=0, **kw)
        ax[0, 1].plot(ii, tab.scatter[k][ok] * 1e-3, **kw)
        ax[1, 0].plot(ii, np.maximum(hop[k][ok], 1e-9), **kw)
        j = int(np.argmin(tm))                 # operating branch: intensities at and above the T minimum
        if len(tm) > j + 1:
            ax[1, 1].plot(tm[j:], tab.scatter[k][ok][j:] * 1e-3, **kw)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xscale("log")
        a.set_xlabel("Intensity per beam (mW cm$^{-2}$)")
    ax[0, 1].set_yscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 1].set_yscale("log")
    ax[0, 0].set_ylabel("Temperature (µK)")
    ax[0, 1].set_ylabel("Scattering rate (10$^3$ s$^{-1}$)")
    ax[1, 0].set_ylabel("Hopping rate (s$^{-1}$)")
    ax[1, 0].set_ylim(bottom=1e-6)
    ax[1, 1].set_xlabel("Temperature (µK)")
    ax[1, 1].set_ylabel("Scattering rate (10$^3$ s$^{-1}$)")
    ax[1, 1].text(0.97, 0.05, "intensity ≥ optimum", transform=ax[1, 1].transAxes, ha="right", fontsize=6,
                  color="#555555")
    ax[0, 0].legend(title="Detuning", fontsize=6, title_fontsize=6, ncol=2)
    ax[1, 0].text(0.03, 0.95, "full depth (19, 16, 12.5 kHz)\nbarriers 268, 192, 115 µK", transform=ax[1, 0].transAxes,
                  va="top", fontsize=6, color="#555555")
    for a, s in zip(ax.flat, "abcd"):
        letter(a, s, -0.16)
    fig.savefig(out)
    plt.close(fig)


def figure_field(tab, out):
    pairs = tab.field_pairs()
    dets = sorted({p[0] for p in pairs})
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    show = [dets[0], dets[-1]] if len(dets) > 1 else dets
    ints = sorted({p[1] for p in pairs})
    icol = ["#56B4E9", "#E69F00", "#CC79A7"]
    for a, d in zip(ax[:2], show):
        for n, i in enumerate(ints):
            for ka in range(3):
                c = tab.field_curve(d, i, ka)
                if c is None:
                    continue
                a.plot(c[0], c[1].mean(axis=1), color=icol[n % 3], ls=AXIS_STYLES[ka], marker="o", ms=2,
                       label=f"{i:g} mW cm$^{{-2}}$" if ka == 0 else (f"B ∥ {'xyz'[ka]}" if n == 0 else None))
        a.set_yscale("log")
        a.set_xlabel("Residual field (mG)")
        a.set_ylabel("Temperature (µK)")
        a.set_title(f"{d:+.0f} MHz", fontsize=7)
    if len(show):
        h, lab = ax[0].get_legend_handles_labels()
        ax[0].legend(h, [l.replace("B ∥ y", "B ∥ y (dashed)").replace("B ∥ z", "B ∥ z (dotted)") for l in lab],
                     fontsize=5.5)
    cols = det_colours(len(tab.detunings))
    bmax = max(abs(np.asarray(r["b_mg"])).max() for r in tab.field_rows)
    for k, d in enumerate(tab.detunings):
        ii = [i for i in ints if (d, i) in pairs]
        if not ii:
            continue
        b2 = np.array([np.mean([tab.doubling_field(d, i, a) for a in range(3)]) for i in ii])
        fin = np.isfinite(b2)
        ax[2].plot(np.array(ii)[fin], b2[fin], color=cols[k], marker="o", ms=3, label=f"{d:+.0f} MHz")
        ax[2].plot(np.array(ii)[~fin], np.full((~fin).sum(), bmax), color=cols[k], marker="^", ms=3, mfc="none",
                   ls="none")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Intensity per beam (mW cm$^{-2}$)")
    ax[2].set_ylabel("Field doubling T (mG)")
    ax[2].legend(fontsize=6)
    for a, s in zip(ax, "abc"):
        letter(a, s)
    fig.savefig(out)
    plt.close(fig)


def main():
    apply()
    tab = T.Tables.from_dir()
    (REPO / "figures").mkdir(exist_ok=True)
    if np.isfinite(tab.scatter).any():
        figure_intensity(tab, REPO / "figures" / "molasses_intensity.png")
        print("wrote figures/molasses_intensity.png")
    if tab.field_rows:
        figure_field(tab, REPO / "figures" / "molasses_field.png")
        print("wrote figures/molasses_field.png")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
