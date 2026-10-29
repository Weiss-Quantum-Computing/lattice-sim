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
import matplotlib.ticker

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
        a.set_xlabel("Residual field (mG)")
        a.set_ylabel("Temperature (µK)")
        a.set_title(f"{d:+g} MHz", fontsize=7)
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
        ax[2].plot(np.array(ii)[fin], b2[fin], color=cols[k], marker="o", ms=3, label=f"{d:+g} MHz")
        ax[2].plot(np.array(ii)[~fin], np.full((~fin).sum(), bmax), color=cols[k], marker="^", ms=3, mfc="none",
                   ls="none")
    ax[2].set_xscale("log")
    ax[2].set_xticks([0.2, 0.5, 1.0, 2.0])
    ax[2].set_xticklabels(["0.2", "0.5", "1", "2"])
    ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].text(0.97, 0.03, "open: T does not double\nwithin the scanned 800 mG", transform=ax[2].transAxes,
               ha="right", va="bottom", fontsize=5.5, color="#555555")
    ax[2].set_xlabel("Intensity per beam (mW cm$^{-2}$)")
    ax[2].set_ylabel("Field doubling T (mG)")
    ax[2].legend(fontsize=6)
    for a, s in zip(ax, "abc"):
        letter(a, s)
    fig.savefig(out)
    plt.close(fig)


def _hops(r, attempt=None):
    """Arrhenius hopping rate (sum over axes, s^-1) of one summary row, tail temperature."""
    t = np.asarray(r["t_uk"], float)
    tt = np.asarray(r.get("t_tail_uk", t), float)
    tt = np.where(np.isfinite(tt), np.maximum(tt, t), t)
    tau = np.asarray(r["tau_e_ms"], float) if attempt is None else 1e3 / attempt
    return float(arrhenius_hop_rate(tt, tau, np.asarray(r["barrier_uk"], float)).sum())


def figure_depth(rows, out):
    """a: temperature against lattice depth; b: hopping rate, counted in the Monte Carlo (markers) and
    Arrhenius from the tail temperature with attempt rate 1/tau_E (solid) or Li's 265 s^-1 (dashed)."""
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.6), constrained_layout=True)
    keys = sorted({(r["detuning"], r["intensity"]) for r in rows})
    cols = ["#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00", "#CC79A7"]
    for n, (d, i) in enumerate(keys):
        rr = sorted([r for r in rows if (r["detuning"], r["intensity"]) == (d, i)], key=lambda r: r["depth_scale"])
        f = np.array([r["depth_scale"] for r in rr])
        lab = f"{d:+g} MHz, {i:g} mW cm$^{{-2}}$"
        ax[0].plot(f, [np.mean(r["t_uk"]) for r in rr], color=cols[n % 6], marker="o", ms=3, label=lab)
        obs = np.array([sum(r["hops_obs_per_s"]) for r in rr])
        ax[1].plot(f[obs > 0], obs[obs > 0], color=cols[n % 6], marker="o", ms=3.5, ls="none")
        ax[1].plot(f, [max(_hops(r), 1e-6) for r in rr], color=cols[n % 6], lw=0.9)
        ax[1].plot(f, [max(_hops(r, T.LI_ATTEMPT_RATE_PER_S), 1e-6) for r in rr], color=cols[n % 6], lw=0.8,
                   ls="--")
    for a in ax:
        a.set_xlabel("Lattice depth (fraction of full)")
        a.set_xscale("log")
    ax[0].set_ylabel("Temperature (µK)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-3, 1e5)
    ax[1].set_ylabel("Hopping rate, x+y+z (s$^{-1}$)")
    ax[0].legend(fontsize=5.5)
    ax[1].text(0.97, 0.95, "markers: counted in the Monte Carlo\nsolid: Arrhenius, Γ$_a$ = 1/τ$_E$\n"
               "dashed: Arrhenius, Γ$_a$ = 265 s$^{-1}$", transform=ax[1].transAxes, ha="right", va="top",
               fontsize=5.5, color="#555555")
    for a, s in zip(ax, "ab"):
        letter(a, s, -0.16)
    fig.savefig(out)
    plt.close(fig)


def figure_lab1026(model_rows, out):
    """The 2026-10-26 hopping-sequence power scans against the model at -25.5 MHz.
    a: loss per image cycle (~0.9 s of molasses); b: model temperature; c: brightness against model R."""
    import csv
    meas = []
    with open(REPO / "data" / "measured" / "molasses_hopping_20261026.csv", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            meas.append({k: float(v) for k, v in row.items()})
    exposure_s = LAB_1026_MOLASSES_S
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    dcol = {-30.5: "#009E73", -25.5: "#0072B2", -20.5: "#D55E00"}
    for d, c in dcol.items():
        m = sorted([r for r in meas if r["detuning_mhz"] == d], key=lambda r: r["rescale"])
        p = np.array([24.0 / r["rescale"] for r in m])
        loss = np.array([r["losses"] / r["atoms_set0"] for r in m])
        err = np.sqrt(loss * (1 - loss) / np.array([r["atoms_set0"] for r in m]))
        ax[0].errorbar(p, loss, err, color=c, marker="o", ms=3, ls="none", capsize=0, label=f"lab {d:+g} MHz")
        ax[2].plot(p, [r["margin_set0"] for r in m], color=c, marker="o", ms=3, ls="none")
    mr = sorted(model_rows, key=lambda r: r["intensity"])
    if mr:
        p = np.array([r["intensity"] / LAB_1026_I24 for r in mr])
        for att, ls, lab in ((None, "-", "model, Γ$_a$ = 1/τ$_E$"), (T.LI_ATTEMPT_RATE_PER_S, "--", "model, Γ$_a$ = 265 s$^{-1}$")):
            h = np.array([_hops(r, att) for r in mr])
            ax[0].plot(p, 1 - np.exp(-h * exposure_s), color="#0072B2", ls=ls, lw=1, label=lab)
        ax[1].plot(p, [np.mean(r["t_uk"]) for r in mr], color="#0072B2", marker="s", ms=3, label="kinetic")
        ax[1].plot(p, [np.mean(np.maximum(r["t_tail_uk"], r["t_uk"])) for r in mr], color="#0072B2", ls=":",
                   marker="^", ms=3, label="tail")
        ax2p = p
    ax[0].set_yscale("log")
    ax[0].set_ylim(5e-3, 1.2)
    ax[0].set_ylabel(f"Loss per image cycle ({exposure_s:g} s molasses)")
    ax[1].set_ylabel("Model temperature, −25.5 MHz (µK)")
    ax[2].set_ylabel("Atom brightness margin (counts)")
    for a in ax:
        a.set_xlabel("Molasses power (rescale 24 = 1)")
    ax[0].legend(fontsize=5.5)
    ax[1].legend(fontsize=6)
    ax[1].text(0.03, 0.97, f"1 = {LAB_1026_I24:g} mW cm$^{{-2}}$ per beam\n(2.2 µW from the fiber)",
               transform=ax[1].transAxes, va="top", fontsize=5.5, color="#555555")
    for a, s in zip(ax, "abc"):
        letter(a, s)
    fig.savefig(out)
    plt.close(fig)


LAB_1026_I24 = 1.46          # as scripts/molasses_scan.py
LAB_1026_MOLASSES_S = 0.9    # 9 exposures x 100 ms between the two image sets


def main():
    apply()
    tab = T.Tables.from_dir()
    (REPO / "figures").mkdir(exist_ok=True)
    depth = T.load_jsonl(T.DATA_DIR / "depth.jsonl")
    keys = {(r["detuning"], r["intensity"]) for r in depth}
    depth += [r for r in T.load_jsonl(T.DATA_DIR / "intensity.jsonl")          # full depth, same settings
              if (r["detuning"], r["intensity"]) in keys and r["depth_scale"] == 1.0 and not any(r["b_mg"])]
    if depth:
        figure_depth(depth, REPO / "figures" / "molasses_depth.png")
        print("wrote figures/molasses_depth.png")
    lab = T.load_jsonl(T.DATA_DIR / "lab1026.jsonl")
    figure_lab1026(lab, REPO / "figures" / "molasses_lab_20261026.png")
    print("wrote figures/molasses_lab_20261026.png")
    if np.isfinite(tab.scatter).any():
        figure_intensity(tab, REPO / "figures" / "molasses_intensity.png")
        print("wrote figures/molasses_intensity.png")
    if tab.field_rows:
        figure_field(tab, REPO / "figures" / "molasses_field.png")
        print("wrote figures/molasses_field.png")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
