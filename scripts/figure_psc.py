"""Figure: projection sideband cooling versus the EO rotation angle.

a  Franck-Condon factor |<3,-3,n-1|4,-4,n>|^2 (x axis) against n and theta;
b  the same for Delta n = -2;
c  AFP transfer of the -2 pulse |4,-4,n> -> |3,-3,n-2> on x at 6 and 12 deg
   (Omega0 = 20 kHz, 3 ms, HSR -6 kHz), with the Franck-Condon factor;
d  outcome of one x step per level: clean (n -> n-1 without OP) for the lab
   loop 1 (12/6 deg) and for 6/6 deg;
e  3D ground-state fraction against cycle: lab sequence and the model optimum
   (per-axis lab curves dashed); f  the last 15 cycles;
g  final 3D ground fraction against the loop-1 angles (loop 2 at 13/12 deg);
h  final 3D ground fraction against the initial temperature (Omega0 = 20 kHz);
i  against the carrier Rabi frequency Omega0 (5 uK).

Needs data/psc_ladders.npz, data/psc_pulses.npz, data/psc_results.json.
Writes figures/psc_cooling.png. Usage: python scripts/figure_psc.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path("scripts").resolve()))

import matplotlib.pyplot as plt  # noqa: E402

from lattice_sim.cooling import psc  # noqa: E402
from lattice_sim.viz.style import COLORS, DOUBLE_COLUMN_IN, apply  # noqa: E402
from psc_analysis import sequence  # noqa: E402

N_SHOW = 30


def label(ax, s):
    ax.text(-0.2, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


if __name__ == "__main__":
    apply()
    ladders = psc.load_ladders()
    tables = dict(np.load(Path("data") / "psc_pulses.npz"))
    res = json.loads(Path("data/psc_results.json").read_text(encoding="utf-8"))
    op = psc.optical_pumping_model(ladders)
    lib = psc.TableLibrary(ladders, tables, 20.0)
    thetas = np.array(sorted(k[1] for k in ladders if k[0] == "x" and isinstance(k[1], float)))
    n = np.arange(N_SHOW)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 6.2))
    gs = fig.add_gridspec(3, 3, hspace=0.65, wspace=0.75)

    # a, b: FC maps
    for col, k in ((0, 1), (1, 2)):
        ax = fig.add_subplot(gs[0, col])
        fc = np.zeros((thetas.size, N_SHOW))
        for i, th in enumerate(thetas):
            a2 = psc.ladder_at(ladders, "x", th).overlap ** 2
            fc[i] = [a2[m - k, m] if m >= k else np.nan for m in n]
        im = ax.pcolormesh(n, thetas, fc, cmap="cividis", vmin=0, vmax=0.4, shading="nearest", rasterized=True)
        for th, ls in ((6, "--"), (12, "-"), (13, ":")):
            ax.axhline(th, color="white", lw=0.6, ls=ls)
        ax.set_xlabel("initial level $n$")
        ax.set_ylabel(r"rotation $\theta$ (deg)")
        ax.set_title(rf"$|\langle 3,-3,n{-k:+d}\,|\,4,-4,n\rangle|^2$, x", fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("FC factor")
        label(ax, "ab"[col])

    # c: AFP transfer of the -2 pulse vs FC
    ax = fig.add_subplot(gs[0, 2])
    nn = np.arange(2, N_SHOW)
    for th, c in ((6.0, COLORS["blue"]), (12.0, COLORS["red"])):
        t = lib.pulse("x", 2, th, -6e3)
        nl = t.shape[0] // 2
        a2 = psc.ladder_at(ladders, "x", th).overlap ** 2
        ax.plot(nn, [t[nl + m - 2, m] for m in nn], "o-", ms=2, color=c, label=f"AFP, {th:g}°")
        ax.plot(nn, [a2[m - 2, m] for m in nn], ":", color=c, label=f"FC, {th:g}°")
    ax.set_xlabel("initial level $n$")
    ax.set_ylabel(r"$P(n\to n-2)$, FC factor")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=5.5, loc="center right")
    label(ax, "c")

    # d: clean fraction per level, x step
    ax = fig.add_subplot(gs[1, 0])
    for (t2, t1), c, nm in (((12.0, 6.0), COLORS["red"], "lab 12/6°"), ((6.0, 6.0), COLORS["blue"], "6/6°")):
        o = psc.step_outcomes(lib, "x", psc.Loop(1, dict.fromkeys(psc.AXES, t2), dict.fromkeys(psc.AXES, t1), -6e3, -4e3))
        ax.plot(n[1:], o["clean"][1:N_SHOW], "o-", ms=2, color=c, label=f"clean, {nm}")
        ax.plot(n, o["pumped"][:N_SHOW], ":", color=c, label=f"pumped, {nm}")
    ax.set_xlabel("initial level $n$")
    ax.set_ylabel("probability per step")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, columnspacing=0.8)
    label(ax, "d")

    # e: ground fraction vs cycle
    ax = fig.add_subplot(gs[1, 1])
    r20 = res["per_omega"]["20"]
    cyc = np.arange(len(r20["lab_history"]["ground_3d"]))
    ax.plot(cyc, r20["lab_history"]["ground_3d"], color=COLORS["red"], label="lab")
    b = r20["best_same"]
    ax.plot(cyc, r20["best_same_history"]["ground_3d"], color=COLORS["blue"],
            label="optimum")
    for a, ls in zip(psc.AXES, ("--", "-.", ":")):
        ax.plot(cyc, r20["lab_history"]["ground"][a], ls, color=COLORS["red"], lw=0.6, label=f"lab, {a}")
    ax.axvline(30, color=COLORS["grey"], lw=0.5)
    ax.set_xlabel("cooling cycle (x, y, z)")
    ax.set_ylabel("ground-state fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5, loc="lower right")
    label(ax, "e")
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(cyc, r20["lab_history"]["ground_3d"], "o-", ms=2, color=COLORS["red"], label="lab")
    ax.plot(cyc, r20["best_same_history"]["ground_3d"], "o-", ms=2, color=COLORS["blue"], label="optimum")
    ax.axvline(30, color=COLORS["grey"], lw=0.5)
    ax.axhline(0.96, color=COLORS["grey"], lw=0.5, ls=":")
    ax.text(34.5, 0.957, "measured (Du)", fontsize=5.5, color=COLORS["grey"], va="top")
    ax.text(30.5, 0.932, "loop 2", fontsize=5.5, color=COLORS["grey"])
    ax.set_xlim(25, 40.5)
    ax.set_ylim(0.93, 1.0)
    ax.set_xlabel("cooling cycle")
    ax.set_ylabel("3D ground fraction")
    ax.legend(fontsize=5.5, loc="upper left")
    label(ax, "f")

    # f: loop-1 scan
    ax = fig.add_subplot(gs[2, 0])
    grid = np.array(res["grid_deg"])
    s1 = np.array(r20["scan_loop1"])
    im = ax.pcolormesh(grid, grid, s1, cmap="cividis", vmin=0.95, vmax=0.99, shading="nearest", rasterized=True)
    ax.plot(6, 12, "o", mfc="none", mec=COLORS["red"], ms=5, label="lab")
    i, j = np.unravel_index(np.argmax(s1), s1.shape)
    ax.plot(grid[j], grid[i], "x", color="white", ms=5, label="optimum")
    ax.set_xlabel(r"loop 1, $\theta_{-1}$ (deg)")
    ax.set_ylabel(r"loop 1, $\theta_{-2}$ (deg)")
    ax.legend(fontsize=5.5, loc="upper right", labelcolor="white")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("3D ground fraction")
    label(ax, "g")

    # g: sensitivity to T and Omega0
    ax = fig.add_subplot(gs[2, 1])
    temps = np.array([3, 4, 5, 6, 7.0])
    for kw, c, nm in (((12, 6, 13, 12), COLORS["red"], "lab"), ((6, 6, 9, 18), COLORS["blue"], "optimum")):
        g = [psc.ground_3d_factorized(psc.simulate_factorized(
            sequence(*kw), lib, op, p0=psc.initial_state(temperature_k=t * 1e-6))) for t in temps]
        ax.plot(temps, g, "o-", ms=2.5, color=c, label=nm)
    ax.axhline(0.96, color=COLORS["grey"], lw=0.5, ls=":")
    ax.set_xlabel(r"initial temperature ($\mu$K)")
    ax.set_ylabel("3D ground fraction")
    ax.legend(fontsize=5.5)
    label(ax, "h")
    ax = fig.add_subplot(gs[2, 2])
    oms = np.array(res["omegas_khz"])
    ax.plot(oms, [res["per_omega"][f"{o:g}"]["lab"] for o in oms], "o-", ms=2.5, color=COLORS["red"], label="lab")
    ax.plot(oms, [res["cross"]["20"][f"{o:g}"] for o in oms], "o-", ms=2.5, color=COLORS["blue"], label="optimum at 20 kHz")
    ax.plot(oms, [res["per_omega"][f"{o:g}"]["best_same_ground"] for o in oms], "s--", ms=2.5, mfc="none",
            color=COLORS["teal"], label=r"optimum at each $\Omega_0$")
    ax.set_xscale("log")
    ax.set_xticks(oms)
    ax.set_xticklabels([f"{o:g}" for o in oms])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    label(ax, "i")
    ax.set_xlabel(r"carrier Rabi frequency $\Omega_0/2\pi$ (kHz)")
    ax.set_ylabel("3D ground fraction")
    ax.legend(fontsize=5.5)

    out = Path("figures") / "psc_cooling.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
