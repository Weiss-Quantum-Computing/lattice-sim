"""Figure and table: carrier and x-sideband gradients across the lattice for
beam offsets and tilts of the x pair (common and differential, in-plane and
vertical, with and without an upstream pivot), against a bias-field
gradient for scale.

Each scenario is evaluated at sites along the fringe direction (x) and
along the static polarization (z, the imaging axis) over +/- 20 um around
the centre, with the y and z pairs nominal. Writes
figures/tilt_gradients.png and prints the gradients in Hz per um.
Usage: python scripts/figure_tilt_gradients.py
"""

import numpy as np

from lattice_sim.spectra.beams import ArmGeometry, PairGeometry, spectra_along
from lattice_sim.spectra.sidebands import collapse_lines, nominal_site
from lattice_sim.viz.style import COLORS, apply

UM = 1e-6
MRAD = 1e-3
PIVOT = 0.30                     # m, a mirror 30 cm upstream of the atoms
TILT_AT_PIVOT = 10 * UM / PIVOT  # 33 urad: moves the beam by 10 um at the atoms
ZEEMAN_HZ_PER_MG = -2452.0


def scenarios():
    arm = ArmGeometry
    return {
        "nominal": PairGeometry(arm(), arm()),
        "both arms +10 um in z": PairGeometry(arm(offset_pol_m=10 * UM), arm(offset_pol_m=10 * UM)),
        "both arms +10 um in x": PairGeometry(arm(offset_fringe_m=10 * UM), arm(offset_fringe_m=10 * UM)),
        "rotating arm +10 um in z": PairGeometry(arm(), arm(offset_pol_m=10 * UM)),
        "diff. vertical tilt 1 mrad, pivot 0": PairGeometry(arm(tilt_vertical_rad=MRAD), arm(tilt_vertical_rad=-MRAD)),
        "diff. vertical tilt 33 urad, pivot 30 cm (+/-10 um)": PairGeometry(
            arm(tilt_vertical_rad=TILT_AT_PIVOT, pivot_m=PIVOT), arm(tilt_vertical_rad=-TILT_AT_PIVOT, pivot_m=PIVOT)),
        "common vertical tilt 1 mrad, pivot 0": PairGeometry(arm(tilt_vertical_rad=MRAD), arm(tilt_vertical_rad=MRAD)),
        "common vertical tilt 33 urad, pivot 30 cm (10 um)": PairGeometry(
            arm(tilt_vertical_rad=TILT_AT_PIVOT, pivot_m=PIVOT), arm(tilt_vertical_rad=TILT_AT_PIVOT, pivot_m=PIVOT)),
        "diff. in-plane tilt 1 mrad (crossing), pivot 0": PairGeometry(arm(tilt_inplane_rad=MRAD), arm(tilt_inplane_rad=-MRAD)),
        "common in-plane tilt 33 urad, pivot 30 cm (10 um)": PairGeometry(
            arm(tilt_inplane_rad=TILT_AT_PIVOT, pivot_m=PIVOT), arm(tilt_inplane_rad=TILT_AT_PIVOT, pivot_m=PIVOT)),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply()
    site = nominal_site()
    pos = np.linspace(-20, 20, 5) * UM
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    colours = list(COLORS.values()) + ["k", "0.5", "#8c564b"]
    for k, (name, geom) in enumerate(scenarios().items()):
        along = {"x": [np.array([p, 0, 0]) for p in pos], "z": [np.array([0, 0, p]) for p in pos]}
        res = {d: spectra_along(site, "x", geom, pts) for d, pts in along.items()}
        c = {d: np.array([s["carrier_shift_hz"] for s in r]) for d, r in res.items()}
        f = {d: np.array([s["axes"]["x"]["f_lower_hz"] for s in r]) for d, r in res.items()}
        centre = res["x"][2]
        ratio = collapse_lines(centre["lines"]).get(("x", 1), (0, 0))[1] / centre["carrier_weight"]
        spacing, _, frame = geom.lattice()
        tilt_deg = np.rad2deg(np.arctan2(frame[0, 2], frame[0, 0]))
        row = dict(name=name, carrier=c["x"][2], f_x=f["x"][2], ratio=ratio, spacing_nm=spacing * 1e9, fringe_tilt_deg=tilt_deg,
                   dc_dx=np.polyfit(pos, c["x"], 1)[0] * UM, dc_dz=np.polyfit(pos, c["z"], 1)[0] * UM,
                   df_dx=np.polyfit(pos, f["x"], 1)[0] * UM, df_dz=np.polyfit(pos, f["z"], 1)[0] * UM)
        rows.append(row)
        print(f"{name:48s} carrier {row['carrier']:+7.1f} Hz  f_x {row['f_x']/1e3:6.3f} kHz  SB {ratio:.3f}  "
              f"a {row['spacing_nm']:.1f} nm  fringe tilt {tilt_deg:+.2f} deg | dcarrier/dx {row['dc_dx']:+6.2f} dz {row['dc_dz']:+6.2f} Hz/um"
              f" | df_x/dx {row['df_dx']:+6.2f} dz {row['df_dz']:+6.2f} Hz/um", flush=True)
        col = colours[k % len(colours)]
        ls = "-" if "tilt" not in name else "--"
        axes[0, 0].plot(pos / UM, c["x"], color=col, ls=ls, label=name)
        axes[0, 1].plot(pos / UM, c["z"], color=col, ls=ls)
        axes[1, 0].plot(pos / UM, f["x"] / 1e3, color=col, ls=ls)
        axes[1, 1].plot(pos / UM, f["z"] / 1e3, color=col, ls=ls)
    b_grad = ZEEMAN_HZ_PER_MG * 0.01                        # 0.1 G/cm = 100 mG per 1e4 um = 0.01 mG/um
    print(f"reference: a 0.1 G/cm bias-field gradient moves the carrier by {b_grad:+.1f} Hz/um "
          f"(1 mG/cm: {b_grad/100:+.2f} Hz/um); no sideband-frequency gradient")
    axes[0, 0].set_ylabel("Carrier shift (Hz)")
    axes[1, 0].set_ylabel("$f_x$, |3,-3> (kHz)")
    axes[1, 0].set_xlabel("Site position along x (um)")
    axes[1, 1].set_xlabel("Site position along z (um)")
    axes[0, 0].set_title("along the fringes (x)", fontsize=7)
    axes[0, 1].set_title("along the imaging axis (z)", fontsize=7)
    axes[0, 0].legend(fontsize=5, loc="lower left")
    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/tilt_gradients.png")
    print("wrote figures/tilt_gradients.png")
    return rows


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main()
