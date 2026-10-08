"""Interactive explorer: how lattice alignment moves the |4,-4> -> |3,-3>
microwave carrier and its Delta n = +/-1 sidebands.

Sliders change one pair at a time (choose x, y or z with the radio buttons):
the transverse centering offset of each arm at the atoms (which lowers that
arm's amplitude by exp(-r^2 / w^2), waist w = 60 um), the crossing
half-angle (which sets the lattice spacing), the depth scale, the rotation
of that pair's EOM, its extinction ratio and handedness. Global sliders set
the bias field (magnitude and angle in the lab x-y plane), the temperature
and the display linewidth. The grey curve is the nominal configuration
(full depth 19 / 16 / 12.5 kHz, 3 % rotation on every pair, ER 3e4, 1.393 G
at 45 deg, 1 uK), the blue curve the current one. The table lists the
carrier shift and the trap frequencies and sideband-to-carrier ratios.

Frequencies are the microwave frequency relative to the free-atom
transition at the bias field. Delta n = +1 (a quantum added) lies at
-f(|3,-3>), Delta n = -1 at +f(|4,-4>).

Usage: python scripts/sideband_explorer.py            # interactive window
       python scripts/sideband_explorer.py --save OUT.png   # nominal spectrum only
"""

import sys
from dataclasses import replace

import numpy as np

from lattice_sim.spectra.sidebands import (AxisConfig, SiteConfig, amplitude_from_offset, collapse_lines,
                                           line_shape, nominal_site, spacing_from_crossing, spectrum)

WAIST_M = 60e-6
F_RANGE_KHZ = 25.0
NOMINAL_HALF_ANGLE_DEG = np.rad2deg(np.arcsin(843.222e-9 / (2 * 4.8374e-6)))


class Explorer:
    def __init__(self):
        self.nominal = nominal_site()
        self.ref = spectrum(self.nominal, max_dn=1)
        self.axis = "x"
        self.state = {ax: dict(static_um=0.0, rotating_um=0.0, dalpha_deg=0.0, depth=1.0, rotation_pct=3.0,
                               log_er=np.log10(3e4), handedness=-1)
                      for ax in "xyz"}
        self.glob = dict(b_gauss=1.39304, b_angle_deg=45.0, temperature_uk=1.0, width_hz=300.0)

    def site(self):
        axes = []
        for cfg in self.nominal.axes:
            s = self.state[cfg.axis]
            axes.append(replace(
                cfg,
                trap_frequency_hz=cfg.trap_frequency_hz * np.sqrt(s["depth"]),
                spacing_m=spacing_from_crossing(np.deg2rad(NOMINAL_HALF_ANGLE_DEG + s["dalpha_deg"])),
                theta=s["rotation_pct"] / 100 * np.pi,
                extinction_ratio=10 ** s["log_er"], handedness=int(s["handedness"]),
                static_amplitude=amplitude_from_offset(s["static_um"] * 1e-6, WAIST_M),
                rotating_amplitude=amplitude_from_offset(s["rotating_um"] * 1e-6, WAIST_M)))
        phi = np.deg2rad(self.glob["b_angle_deg"])
        b = self.glob["b_gauss"] * np.array([np.cos(phi), np.sin(phi), 0.0])
        return SiteConfig(axes, b, self.glob["temperature_uk"] * 1e-6)

    def table(self, sp):
        rows = [f"carrier shift {sp['carrier_shift_hz']:+8.1f} Hz  (nominal {self.ref['carrier_shift_hz']:+.1f})"]
        lines = collapse_lines(sp["lines"])
        for ax in "xyz":
            a = sp["axes"][ax]
            r_plus = lines.get((ax, 1), (0, 0))[1] / sp["carrier_weight"]
            r_minus = lines.get((ax, -1), (0, 0))[1] / sp["carrier_weight"]
            rows.append(f"{ax}: f(3,-3) {a['f_lower_hz']/1e3:6.2f} kHz  f(4,-4) {a['f_upper_hz']/1e3:6.2f} kHz  "
                        f"dx {a['displacement_m']*1e9:+6.1f} nm  SB/carrier +1 {r_plus:.3f}  -1 {r_minus:.3f}")
        return "\n".join(rows)


def curve(sp, width_hz):
    f = np.linspace(-F_RANGE_KHZ, F_RANGE_KHZ, 4001) * 1e3
    return f / 1e3, line_shape(f, sp["lines"], sp["carrier_weight"], width_hz)


def run_interactive():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    ex = Explorer()
    fig = plt.figure(figsize=(12, 7))
    ax_sp = fig.add_axes([0.06, 0.40, 0.56, 0.52])
    ax_txt = fig.add_axes([0.06, 0.05, 0.56, 0.28])
    ax_txt.axis("off")
    f, y_ref = curve(ex.ref, ex.glob["width_hz"])
    (l_ref,) = ax_sp.plot(f, y_ref, color="0.6", lw=1.0, label="nominal")
    (l_cur,) = ax_sp.plot(f, y_ref, color="#1f5fa8", lw=1.2, label="current")
    ax_sp.set_xlabel("Microwave detuning from the free-atom |4,-4> -> |3,-3> line (kHz)")
    ax_sp.set_ylabel("Transition strength (carrier = 1 at T = 0)")
    ax_sp.set_yscale("log")
    ax_sp.set_ylim(1e-5, 2)
    ax_sp.legend(loc="upper right")
    txt = ax_txt.text(0, 1, ex.table(ex.ref), va="top", family="monospace", fontsize=9)

    sliders = {}
    x0, w, h, dy = 0.70, 0.25, 0.022, 0.034
    y = 0.93

    def add(name, label, lo, hi, val, fmt="%.2f"):
        nonlocal y
        ax = fig.add_axes([x0, y, w, h])
        sliders[name] = Slider(ax, label, lo, hi, valinit=val, valfmt=fmt)
        y -= dy

    ax_radio = fig.add_axes([x0, y - 0.06, 0.08, 0.08])
    radio = RadioButtons(ax_radio, ("x", "y", "z"), active=0)
    fig.text(x0 + 0.09, y - 0.01, "pair edited by the\nsliders below", fontsize=8, va="top")
    y -= 0.10
    add("static_um", "static arm offset (um)", 0, 40, 0, "%.1f")
    add("rotating_um", "rotating arm offset (um)", 0, 40, 0, "%.1f")
    add("dalpha_deg", "crossing half-angle change (deg)", -0.5, 0.5, 0, "%.3f")
    add("depth", "depth scale", 0.3, 1.5, 1.0)
    add("rotation_pct", "EOM rotation (% of 180 deg)", 0, 10, 3.0, "%.2f")
    add("log_er", "log10 extinction ratio", 2, 5, np.log10(3e4))
    add("handedness", "handedness (-1 / +1)", -1, 1, -1, "%.0f")
    y -= 0.02
    add("b_gauss", "bias field (G)", 1.2, 1.6, 1.39304, "%.4f")
    add("b_angle_deg", "bias-field angle in x-y (deg)", 0, 90, 45, "%.1f")
    add("temperature_uk", "temperature (uK)", 0, 30, 1, "%.1f")
    add("width_hz", "display linewidth (Hz)", 50, 2000, 300, "%.0f")

    per_axis = ("static_um", "rotating_um", "dalpha_deg", "depth", "rotation_pct", "log_er", "handedness")

    def refresh(_=None):
        for name in per_axis:
            v = sliders[name].val
            ex.state[ex.axis][name] = float(np.sign(v) if name == "handedness" and v != 0 else v)
        for name in ("b_gauss", "b_angle_deg", "temperature_uk", "width_hz"):
            ex.glob[name] = float(sliders[name].val)
        sp = spectrum(ex.site(), max_dn=1)
        f, yc = curve(sp, ex.glob["width_hz"])
        _, yr = curve(ex.ref, ex.glob["width_hz"])
        l_cur.set_ydata(yc)
        l_ref.set_ydata(yr)
        txt.set_text(ex.table(sp))
        fig.canvas.draw_idle()

    def choose_axis(label):
        ex.axis = label
        s = ex.state[label]
        for name in per_axis:
            sliders[name].eventson = False
            sliders[name].set_val(s[name])
            sliders[name].eventson = True
        refresh()

    for s in sliders.values():
        s.on_changed(refresh)
    radio.on_clicked(choose_axis)
    fig.suptitle("Lattice alignment and the |4,-4> -> |3,-3> sideband spectrum (1D per pair; see module docstring)")
    plt.show()


def save_nominal(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ex = Explorer()
    f, y = curve(ex.ref, 300.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, y)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2)
    ax.set_xlabel("Microwave detuning (kHz)")
    ax.set_ylabel("Transition strength")
    ax.set_title("nominal configuration")
    fig.text(0.12, -0.25, ex.table(ex.ref), family="monospace", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(ex.table(ex.ref))


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    if "--save" in sys.argv:
        save_nominal(sys.argv[sys.argv.index("--save") + 1])
    else:
        run_interactive()
