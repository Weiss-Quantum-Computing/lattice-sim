"""Figure: first-principles trap position and frequency minus the ER316
calibration files, across the mid-rotation region, for the transported
pair |3,-3> / |4,-3> under four polarization hypotheses (pure linear, the
measured extinction profile with either handedness, and ER 316).

The pure-linear and measured-profile models differ from the calibration by
a smooth dipole of +/-50-80 nm in x0 and +/-300-500 Hz in f on either side
of pi/2 (opposite signs for the two states), while matching the pi/2
frequency; the ER 316 model matches away from pi/2 and misses the pi/2
frequency by 0.75 kHz. The calibration therefore behaves like ER 316 away
from pi/2 and like pure linear light at pi/2, which is what a generator
that applied the ellipticity to the vector shift but not to the quadrature
scalar lattice would produce. Fast ramps (< 3 ms) are sensitive to this
shape difference (scripts/score_suite.py).

Writes figures/model_vs_calibration.png. Usage: python scripts/figure_model_vs_calibration.py
"""

import numpy as np

from score_suite import B_Q, ER_CSV, STATES, calibration_scenarios
from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.viz.style import COLORS, apply


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply()
    profile = ExtinctionProfile.from_rows(read_rampol_er(ER_CSV), 1, "away")
    cal = calibration_scenarios()
    th = np.linspace(0.3, 0.7, 161) * np.pi
    variants = {"pure linear": (np.inf, -1), "measured ER, h = -1": (profile, -1),
                "measured ER, h = +1": (profile, 1), "ER 316, h = -1": (316.0, -1)}
    colours = [COLORS["blue"], COLORS["red"], COLORS["teal"], COLORS["grey"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for (name, (er, h)), colour in zip(variants.items(), colours):
        model = StatePotential(LatticePair(extinction_ratio=er, handedness=h), B_Q)
        for j, (key, state) in enumerate(STATES.items()):
            x0, f = model.track(state, th)
            cx, cf = cal[key]
            axes[0, j].plot(th / np.pi, (np.abs(x0) - cx(th)) * 1e9, color=colour, label=name)
            axes[1, j].plot(th / np.pi, f - cf(th), color=colour)
            print(f"{name:22s} {state}: max |dx0| {np.abs(np.abs(x0) - cx(th)).max() * 1e9:4.0f} nm, "
                  f"max |df| {np.abs(f - cf(th)).max():4.0f} Hz, f(pi/2) {f[80] / 1e3:.2f} kHz "
                  f"(calibration {cf(np.pi / 2) / 1e3:.2f})", flush=True)
    for j, state in enumerate(STATES.values()):
        axes[0, j].set_title(f"|{state[0]},{state[1]}>: model minus ER316 calibration", fontsize=7)
        axes[0, j].set_ylabel("$\\Delta x_0$ (nm)")
        axes[1, j].set_ylabel("$\\Delta f$ (Hz)")
        axes[1, j].set_xlabel("Rotation angle $\\theta/\\pi$")
        for ax in axes[:, j]:
            ax.axhline(0, color="k", lw=0.4)
    axes[0, 0].legend(fontsize=6)
    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.text(-0.18, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/model_vs_calibration.png")
    print("wrote figures/model_vs_calibration.png")


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main()
