"""Figure: sensitivity of the |4,-4> -> |3,-3> carrier and sidebands to
lattice alignment, one parameter at a time around the nominal sideband
configuration (full depth, 3 % rotation, ER 3e4, 1.393 G at 45 deg, 5 uK).

a  the nominal spectrum with the six first-order sidebands labelled;
b  x trap frequency and x sideband ratio against the centering offset of
   the x pair (both arms together: depth; rotating arm alone: imbalance);
c  the same against the crossing half-angle;
d  carrier shift and x sideband ratio against the extinction ratio of the
   x pair, both handedness values, and against the EOM rotation;
e  carrier shift against the bias-field angle and magnitude.

Writes figures/sidebands_alignment.png. Usage: python scripts/figure_sidebands.py
"""

import numpy as np

from lattice_sim.spectra.sidebands import collapse_lines, line_shape, nominal_site, spectrum, with_alignment
from lattice_sim.viz.style import COLORS, apply

NOMINAL_HALF_ANGLE_DEG = np.rad2deg(np.arcsin(843.222e-9 / (2 * 4.8374e-6)))


def ratios(sp, axis="x"):
    lines = collapse_lines(sp["lines"])
    return (lines.get((axis, 1), (0, 0))[1] / sp["carrier_weight"],
            lines.get((axis, -1), (0, 0))[1] / sp["carrier_weight"])


def scan(site_of, values):
    f_l, f_u, shift, r_plus = [], [], [], []
    for v in values:
        sp = spectrum(site_of(v), max_dn=1)
        f_l.append(sp["axes"]["x"]["f_lower_hz"])
        f_u.append(sp["axes"]["x"]["f_upper_hz"])
        shift.append(sp["carrier_shift_hz"])
        r_plus.append(ratios(sp)[0])
    return np.array(f_l), np.array(f_u), np.array(shift), np.array(r_plus)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply()
    nominal = nominal_site()
    ref = spectrum(nominal, max_dn=1)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    (a, b, c), (d, e, f_) = axes

    # a: nominal spectrum
    freq = np.linspace(-25e3, 25e3, 4001)
    a.plot(freq / 1e3, line_shape(freq, ref["lines"], ref["carrier_weight"], 200.0), color=COLORS["blue"])
    a.set_yscale("log")
    a.set_ylim(1e-5, 2)
    for k, ((ax_name, dn), (off, wt)) in enumerate(sorted(collapse_lines(ref["lines"]).items(), key=lambda i: i[1][0])):
        if abs(dn) == 1 and wt > 1e-5:
            a.annotate(f"{ax_name}{'+' if dn > 0 else '-'}", (off / 1e3, wt / ref["carrier_weight"]),
                       textcoords="offset points", xytext=(0, 4 + 7 * (k % 3)), ha="center", fontsize=6)
    a.set_xlabel("Microwave detuning (kHz)")
    a.set_ylabel("Strength / carrier")
    a.text(0.5, 0.03, f"nominal: carrier {ref['carrier_shift_hz']:+.0f} Hz from the free atom", transform=a.transAxes,
           ha="center", va="bottom", fontsize=6)

    # b: centering offset
    offs = np.linspace(0, 40, 9)
    fl1, _, _, r1 = scan(lambda r: with_alignment(nominal, "x", r * 1e-6, r * 1e-6), offs)
    fl2, _, _, r2 = scan(lambda r: with_alignment(nominal, "x", 0.0, r * 1e-6), offs)
    b.plot(offs, fl1 / 1e3, color=COLORS["blue"], label="both arms off-centre")
    b.plot(offs, fl2 / 1e3, color=COLORS["red"], label="rotating arm only")
    b.set_xlabel("x-pair centering offset (um)")
    b.set_ylabel("x trap frequency, |3,-3> (kHz)")
    b.legend(fontsize=6)
    b2 = b.twinx()
    b2.plot(offs, r1, color=COLORS["blue"], ls="--")
    b2.plot(offs, r2, color=COLORS["red"], ls="--")
    b2.set_ylabel("x sideband / carrier (dashed)")

    # c: crossing angle
    dal = np.linspace(-0.5, 0.5, 9)
    fl3, _, sh3, r3 = scan(lambda da: with_alignment(nominal, "x", crossing_half_angle_deg=NOMINAL_HALF_ANGLE_DEG + da), dal)
    c.plot(dal, fl3 / 1e3, color=COLORS["blue"])
    c.set_xlabel("x crossing half-angle change (deg)")
    c.set_ylabel("x trap frequency, |3,-3> (kHz)")
    c2 = c.twinx()
    c2.plot(dal, r3, color=COLORS["blue"], ls="--")
    c2.set_ylabel("x sideband / carrier (dashed)")

    # d: extinction ratio, both handedness, and rotation
    ers = np.logspace(2, 5, 10)
    for h, col in ((-1, COLORS["blue"]), (1, COLORS["red"])):
        _, _, sh, r = scan(lambda er, h=h: with_alignment(nominal, "x", extinction_ratio=er, handedness=h), ers)
        d.plot(ers, sh, color=col, label=f"h = {h:+d}")
        d2 = d.twinx() if h == -1 else d2
        d2.plot(ers, r, color=col, ls="--")
    d.set_xscale("log")
    d.set_xlabel("x-pair extinction ratio")
    d.set_ylabel("Carrier shift (Hz)")
    d2.set_ylabel("x sideband / carrier (dashed)")
    d.legend(fontsize=6)

    # e: rotation
    rots = np.linspace(0, 10, 11)
    _, _, sh5, r5 = scan(lambda p: with_alignment(nominal, "x", theta=p / 100 * np.pi), rots)
    e.plot(rots, sh5, color=COLORS["blue"])
    e.set_xlabel("x EOM rotation (% of 180 deg)")
    e.set_ylabel("Carrier shift (Hz)")
    e2 = e.twinx()
    e2.plot(rots, np.maximum(r5, 1e-7), color=COLORS["blue"], ls="--")
    e2.set_yscale("log")
    e2.set_ylim(1e-6, 2)
    e2.set_ylabel("x sideband / carrier (dashed)")

    # f: bias field angle (the magnitude enters as the Zeeman slope, -2.452 kHz/mG)
    angs = np.linspace(0, 90, 10)
    from dataclasses import replace
    sh6 = [spectrum(replace(nominal, b_lab_gauss=1.39304 * np.array([np.cos(np.deg2rad(p)), np.sin(np.deg2rad(p)), 0])),
                    max_dn=1)["carrier_shift_hz"] for p in angs]
    r6 = [ratios(spectrum(replace(nominal, b_lab_gauss=1.39304 * np.array([np.cos(np.deg2rad(p)), np.sin(np.deg2rad(p)), 0])),
                          max_dn=1))[0] for p in angs]
    f_.plot(angs, sh6, color=COLORS["blue"])
    f_.set_xlabel("Bias-field angle in x-y (deg)")
    f_.set_ylabel("Carrier shift (Hz)")
    f2 = f_.twinx()
    f2.plot(angs, r6, color=COLORS["blue"], ls="--")
    f2.set_ylabel("x sideband / carrier (dashed)")
    f_.text(0.03, 0.05, "1.393 G; the Zeeman term\nmoves the carrier by\n-2.45 kHz per mG", transform=f_.transAxes,
            fontsize=6)

    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.text(-0.32, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/sidebands_alignment.png")
    print("wrote figures/sidebands_alignment.png")


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main()
