"""Interactive explorer: how lattice alignment moves the |4,-4> -> |3,-3>
microwave carrier and its Delta n = +/-1 sidebands at one lattice site.

Why there are sidebands at all. The microwave photon carries no useful
momentum, so a sideband appears only when |4,-4> and |3,-3> see different
wells: displaced (a small rotation of the EOM arm's polarization, an
ellipticity, an arm imbalance) or of different curvature. A sideband's
position is a trap frequency, its height measures the well displacement,
and the carrier's shift is the differential light shift plus zero point.
Each pair is a 1D problem (``lattice_sim.spectra.sidebands``); the three
are combined as independent problems.

Window. Left: the settings, each with a one-line explanation. The pair
chosen at the top owns the "beam alignment" and "EOM arm" groups (each pair
keeps its own values); the last group is shared. Right: tabs

* Spectrum - the current settings against the nominal ones (full depth
  19 / 16 / 12.5 kHz, polarization angle 5.4 deg on every pair, ER 3e4,
  h = -1, 1.39304 G at 45 deg in x-y, 1 uK), on the detuning from the
  free-atom line at the set bias field;
* Wells - the two states' potentials, vibrational levels and ground-state
  densities along the edited pair's fringes, i.e. where the sidebands come from;
* Scan - one setting swept over a range: carrier shift, trap frequencies and
  sideband-to-carrier ratios of all three pairs.

Below the tabs: carrier shift, the free-atom line, a table per pair (current
and nominal), and the gradients across the lattice when an arm of the edited
pair is off-centre. Calculations run on a worker thread; the vibrational
levels of each pair are cached, so temperature and linewidth changes and
changes to one pair recompute nothing or only that pair.

Usage: python scripts/sideband_explorer.py                 # interactive window
       python scripts/sideband_explorer.py --save OUT.png    # nominal spectrum, no window
"""

import math
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lattice_sim.atom.ground_state import eigenlevels
from lattice_sim.constants import H, KB
from lattice_sim.spectra.beams import NOMINAL_HALF_ANGLE, ArmGeometry, PairGeometry, gradients
from lattice_sim.spectra.sidebands import (LOWER, UPPER, SiteConfig, amplitude_from_offset, collapse_lines,
                                           line_shape, nominal_site, spacing_from_crossing, spectrum,
                                           thermal_weights)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PAIRS = "xyz"
WAIST_M = 60e-6
NOMINAL_ALPHA_DEG = float(np.rad2deg(NOMINAL_HALF_ANGLE))
GRADIENT_HALF_SPAN_M = 10e-6
MAX_LEVELS = 60                  # vibrational levels kept per pair at the highest temperature
DROPPED_POPULATION = 1e-3        # thermal population allowed above the last kept level


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    unit: str
    lo: float
    hi: float
    default: float
    digits: int
    help: str
    log: bool = False            # slider moves log10 of the value

    def fmt(self, v):
        if self.log:
            mantissa, exponent = f"{v:.1e}".split("e")
            return f"{mantissa}e{int(exponent)}"
        return f"{v:.{self.digits}f}"


PAIR_PARAMS = [
    Param("static_um", "Static arm offset", "µm", -40, 40, 0.0, 1,
          "Distance of the static arm's beam axis from the atoms. That arm's field at the atoms falls as "
          f"exp(−r²/w²), w = {WAIST_M * 1e6:.0f} µm: the lattice gets shallower and the arms unbalanced."),
    Param("rotating_um", "Rotating (EOM) arm offset", "µm", -40, 40, 0.0, 1,
          "The same for the EOM arm. Offsetting one arm only unbalances the two arms; offsetting both "
          "equally lowers the depth with the arms still balanced."),
    Param("dalpha_deg", "Crossing half-angle change", "deg", -0.5, 0.5, 0.0, 3,
          f"Change of the half-angle between the two arms (nominal {NOMINAL_ALPHA_DEG:.2f} deg). It sets the "
          "spacing λ / (2 sin α) and, at fixed intensity, the trap frequency (∝ 1/spacing)."),
    Param("depth", "Lattice depth", "x full", 0.3, 1.5, 1.0, 2,
          "Intensity relative to full depth. Trap frequencies scale as √depth; full depth is "
          "19.0 / 16.0 / 12.5 kHz for the x / y / z pairs."),
    Param("theta_deg", "EOM arm polarization angle", "deg", 0.0, 18.0, 5.4, 2,
          "Rotation of the EOM arm's linear polarization away from the static arm's (180 deg at V_π). A small "
          "angle makes the light's polarization vary across the well, so the vector light shift pushes "
          "|4,−4⟩ and |3,−3⟩ apart; the separated wells give the sidebands."),
    Param("er", "EOM arm extinction ratio", "", 1e2, 1e5, 3e4, 0,
          "Intensity ratio of the major to the minor polarization axis of the EOM arm. The minor component, "
          "in quadrature, adds a fictitious magnetic field (vector light shift): it moves the carrier "
          "and changes the well separation.", log=True),
]
GLOBAL_PARAMS = [
    Param("b_gauss", "Bias field magnitude", "G", 1.2, 1.6, 1.39304, 4,
          "1.393 G is the magic field of |3,−1⟩ ↔ |4,+1⟩. The free-atom |4,−4⟩ → |3,−3⟩ line moves with "
          "the field (given below the plot); the spectrum is drawn relative to it, so the plot shows only "
          "the light-shift part."),
    Param("b_angle_deg", "Bias field angle in the x-y plane", "deg", 0.0, 90.0, 45.0, 1,
          "Direction of the bias field in the horizontal plane, from the x axis. Each pair's fictitious field "
          "points along its beams; its component along the bias field shifts the levels."),
    Param("temperature_uk", "Temperature", "µK", 0.0, 5.0, 1.0, 2,
          "The initial vibrational levels are thermally populated (mean occupation in the table). Lines "
          "that remove a quantum (Δn = -1) need n ≥ 1, so they grow with temperature."),
    Param("width_hz", "Display linewidth", "Hz", 50.0, 2000.0, 300.0, 0,
          "Lorentzian half-width used only to draw the lines. It changes no computed number."),
]
PARAMS = {p.key: p for p in PAIR_PARAMS + GLOBAL_PARAMS}
DISPLAY_ONLY = {"width_hz"}


def default_settings():
    pair = {p.key: p.default for p in PAIR_PARAMS}
    pair["handedness"] = -1
    return dict(pairs={ax: dict(pair) for ax in PAIRS},
                glob={p.key: p.default for p in GLOBAL_PARAMS},
                edited="x", offset_dir="pol")


def copy_settings(s):
    return dict(pairs={ax: dict(v) for ax, v in s["pairs"].items()}, glob=dict(s["glob"]),
                edited=s["edited"], offset_dir=s["offset_dir"])


# ------------------------------------------------------------------ model
_NOMINAL = nominal_site()


def n_levels_for(temperature_k, trap_hz):
    """Vibrational levels per pair so that at most DROPPED_POPULATION of the
    thermal population lies above the last kept level (6 at least)."""
    if temperature_k <= 0:
        return 6
    r = math.exp(-H * min(trap_hz) / (KB * temperature_k))
    if r <= 0:
        return 6
    return int(min(MAX_LEVELS, max(6, math.ceil(math.log(DROPPED_POPULATION) / math.log(r)) + 1)))


def site_from(settings):
    """SiteConfig of the settings (all three pairs, field, temperature)."""
    axes = []
    for cfg in _NOMINAL.axes:
        s = settings["pairs"][cfg.axis]
        spacing = spacing_from_crossing(np.deg2rad(NOMINAL_ALPHA_DEG + s["dalpha_deg"]))
        axes.append(replace(
            cfg,
            # the crossing angle changes the spacing at fixed intensity: f ~ 1 / spacing
            trap_frequency_hz=cfg.trap_frequency_hz * np.sqrt(s["depth"]) * cfg.spacing_m / spacing,
            spacing_m=spacing,
            theta=np.deg2rad(s["theta_deg"]),
            extinction_ratio=float(s["er"]), handedness=int(s["handedness"]),
            static_amplitude=amplitude_from_offset(s["static_um"] * 1e-6, WAIST_M),
            rotating_amplitude=amplitude_from_offset(s["rotating_um"] * 1e-6, WAIST_M)))
    g = settings["glob"]
    phi = np.deg2rad(g["b_angle_deg"])
    b = g["b_gauss"] * np.array([np.cos(phi), np.sin(phi), 0.0])
    t = g["temperature_uk"] * 1e-6
    return SiteConfig(axes, b, t, n_levels=n_levels_for(t, [c.trap_frequency_hz for c in axes]))


def compute(settings, cache=None):
    """Spectrum of the settings, with timing and the level count."""
    t0 = time.perf_counter()
    site = site_from(settings)
    sp = spectrum(site, max_dn=1, cache=cache)
    return dict(settings=copy_settings(settings), sp=sp, n_levels=site.n_levels,
                temperature_k=site.temperature_k, seconds=time.perf_counter() - t0)


def needs_gradients(settings):
    s = settings["pairs"][settings["edited"]]
    return s["static_um"] != 0 or s["rotating_um"] != 0


def compute_gradients(settings, cache=None):
    """Carrier and f(|3,−3⟩) gradients across the lattice (Hz/µm) from the
    edited pair's arm offsets, taken along the chosen direction."""
    axis = settings["edited"]
    s = settings["pairs"][axis]
    key = "offset_fringe_m" if settings["offset_dir"] == "fringe" else "offset_pol_m"
    geom = PairGeometry(ArmGeometry(**{key: s["static_um"] * 1e-6}),
                        ArmGeometry(**{key: s["rotating_um"] * 1e-6}))
    site = site_from(settings)
    base = [replace(cfg, static_amplitude=1.0, rotating_amplitude=1.0) if cfg.axis == axis else cfg
            for cfg in site.axes]
    g = gradients(replace(site, axes=base), axis, geom, GRADIENT_HALF_SPAN_M, cache=cache)
    return {k: v * 1e-6 for k, v in g.items() if k != "centre"}


def zeeman_slope_hz_per_g(b_gauss, angle_deg=45.0, step_g=1e-4):
    """d(free-atom |4,−4⟩ → |3,−3⟩ frequency) / d|B|, Hz per gauss."""
    phi = np.deg2rad(angle_deg)
    u = np.array([np.cos(phi), np.sin(phi), 0.0])

    def line(b):
        lv = eigenlevels(b * u)
        return lv[UPPER][0] - lv[LOWER][0]
    return (line(b_gauss + step_g) - line(b_gauss - step_g)) / (2 * step_g)


def sideband_ratios(sp, axis):
    lines = collapse_lines(sp["lines"], axis)
    w = sp["carrier_weight"]
    return lines.get((axis, 1), (0.0, 0.0))[1] / w, lines.get((axis, -1), (0.0, 0.0))[1] / w


def mean_occupation(sp, axis):
    w = sp["axes"][axis]["thermal"]
    return float(np.sum(np.arange(w.size) * w))


def scan(settings, key, values, cache=None, progress=None, cancelled=lambda: False):
    """Sweep one setting (of the edited pair, or a shared one) over values."""
    rows = []
    for i, v in enumerate(values):
        if cancelled():
            return None
        s = copy_settings(settings)
        if key in s["glob"]:
            s["glob"][key] = float(v)
        else:
            s["pairs"][s["edited"]][key] = float(v)
        sp = compute(s, cache)["sp"]
        rows.append(dict(carrier=sp["carrier_shift_hz"],
                         **{f"f_{ax}": sp["axes"][ax]["f_lower_hz"] for ax in PAIRS},
                         **{f"r_{ax}": sideband_ratios(sp, ax)[0] for ax in PAIRS}))
        if progress:
            progress(i + 1, len(values))
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(values=np.asarray(values, float), key=key, edited=settings["edited"],
               current=(settings["glob"].get(key) if key in settings["glob"]
                        else settings["pairs"][settings["edited"]][key]))
    return out


# --------------------------------------------------------------- drawing
CURRENT_COLOUR = "#1f77b4"
NOMINAL_COLOUR = "#9a9a9a"
PAIR_COLOURS = {"x": "#1f77b4", "y": "#d62728", "z": "#2e7d32"}
RATIO_FLOOR = 1e-6
STATE_COLOURS = {UPPER: "#1f77b4", LOWER: "#d62728"}
STATE_LABELS = {UPPER: "|4,−4⟩", LOWER: "|3,−3⟩"}


def spectrum_curve(sp, f_hz, width_hz):
    """Line strength on a detuning axis measured from the free-atom line."""
    return line_shape(np.asarray(f_hz) - sp["carrier_shift_hz"], sp["lines"], sp["carrier_weight"], width_hz)


def _note(ax, text, where="top-left"):
    x = 0.01 if "left" in where else 0.99
    y = 0.98 if "top" in where else 0.02
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7, color="#666666",
            ha="left" if "left" in where else "right", va="top" if "top" in where else "bottom", zorder=5)


def _finish(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    log = ax.get_yscale() == "log"
    ax.grid(True, alpha=0.3, which="both" if log else "major")
    if log:
        ax.grid(True, which="minor", alpha=0.12)


def draw_spectrum(fig, res, ref, width_hz, log=True, show_nominal=True):
    fig.clear()
    ax = fig.add_subplot(111)
    sp, ref = res["sp"], ref["sp"]
    f_max = 1.25 * max(a[k] for s in (sp, ref) for a in s["axes"].values() for k in ("f_upper_hz", "f_lower_hz"))
    f = np.linspace(-f_max, f_max, 8001)
    y = spectrum_curve(sp, f, width_hz)
    if show_nominal:
        ax.plot(f / 1e3, spectrum_curve(ref, f, width_hz), color=NOMINAL_COLOUR, lw=1.0, label="nominal settings")
    ax.plot(f / 1e3, y, color=CURRENT_COLOUR, lw=1.2, label="current settings")
    floor = 1e-5
    if log:
        ax.set_yscale("log")
        ax.set_ylim(floor, 3.0)
    else:
        ax.set_ylim(0, 1.12 * float(y.max()))
    ax.set_xlim(f[0] / 1e3, f[-1] / 1e3)
    for (pair, dn), (off, wt) in sorted(collapse_lines(sp["lines"]).items()):
        if abs(dn) != 1 or wt / sp["carrier_weight"] < (3 * floor if log else 0.01):
            continue
        fx = sp["carrier_shift_hz"] + off
        yy = float(spectrum_curve(sp, [fx], width_hz)[0])
        ax.annotate(f"{pair} {'+1' if dn > 0 else '−1'}", (fx / 1e3, yy), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=PAIR_COLOURS[pair])
    ax.annotate("carrier", (sp["carrier_shift_hz"] / 1e3, float(spectrum_curve(sp, [sp["carrier_shift_hz"]],
                                                                                width_hz)[0])),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    _finish(ax, "Microwave detuning from the free-atom |4,−4⟩ → |3,−3⟩ line at the set field (kHz)",
            "Line strength (undisplaced carrier at T = 0 = 1)",
            "Microwave spectrum |4,−4⟩ → |3,−3⟩ at one lattice site")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    _note(ax, f"T = {res['temperature_k'] * 1e6:.2f} µK, {res['n_levels']} levels per pair\n"
              f"line half-width {width_hz:.0f} Hz (display only)\n"
              "labels: pair, quanta added (Δn)")


def draw_wells(fig, res, pair):
    fig.clear()
    a, b = fig.subplots(2, 1)
    sp = res["sp"]
    data = sp["axes"][pair]
    f_top = 0.0
    for state in (UPPER, LOWER):
        d = data["levels"][state]
        x_nm = d["x"] * 1e9
        dx_nm = x_nm[1] - x_nm[0]
        v = d["v_hz"] - d["v_hz"].min()
        sel = np.abs(x_nm - d["x0_m"] * 1e9) < 320
        col = STATE_COLOURS[state]
        a.plot(x_nm[sel], v[sel] / 1e3, color=col, lw=1.1, label=STATE_LABELS[state])
        for e in d["energies_hz"][:6] - d["v_hz"].min():
            inside = np.where(sel & (v <= e))[0]
            if inside.size:
                a.hlines(e / 1e3, x_nm[inside[0]], x_nm[inside[-1]], color=col, lw=0.7, alpha=0.75)
        f_top = max(f_top, (d["energies_hz"][min(5, d["energies_hz"].size - 1)] - d["v_hz"].min()) * 1.15)
        b.plot(x_nm[sel], d["psi"][0][sel] ** 2 / dx_nm, color=col, lw=1.1, label=STATE_LABELS[state] + ", n = 0")
        b.axvline(d["x0_m"] * 1e9, color=col, lw=0.8, ls="--")
    a.set_ylim(0, f_top / 1e3)
    xlabel = f"Position along the {pair}-pair fringe direction (nm)"
    _finish(a, xlabel, "Potential above its minimum (kHz)",
            f"Wells of the {pair} pair: potential and vibrational levels")
    _finish(b, xlabel, "Probability density (1/nm)", f"Vibrational ground states of the {pair} pair")
    a.legend(loc="upper right", fontsize=7, framealpha=0.85)
    b.legend(loc="upper right", fontsize=7, framealpha=0.85)
    _note(a, f"f = {data['f_upper_hz'] / 1e3:.2f} kHz (|4,−4⟩), {data['f_lower_hz'] / 1e3:.2f} kHz (|3,−3⟩)\n"
             "each state drawn from its own minimum")
    _note(b, f"centre separation x(|3,−3⟩) - x(|4,−4⟩) = {data['displacement_m'] * 1e9:+.1f} nm\n"
             "dashed: centre of each density")


def scan_label(key, edited):
    p = PARAMS[key]
    who = "" if key in {q.key for q in GLOBAL_PARAMS} else f"{edited} pair: "
    label = p.label if (not who or p.label[:2].isupper()) else p.label[0].lower() + p.label[1:]
    return f"{who}{label}" + (f" ({p.unit})" if p.unit else "")


def draw_scan(fig, sc):
    fig.clear()
    if sc is None:
        fig.text(0.5, 0.5, "Choose a setting and a range above, then Run scan.", ha="center", va="center",
                 color="#666666")
        return
    a, b, c = fig.subplots(1, 3)
    x = sc["values"]
    xlabel = scan_label(sc["key"], sc["edited"])
    a.plot(x, sc["carrier"], color="#222222", marker="o", ms=2.5, lw=1.0)
    for pair in PAIRS:
        b.plot(x, sc[f"f_{pair}"] / 1e3, color=PAIR_COLOURS[pair], marker="o", ms=2.5, lw=1.0, label=f"{pair} pair")
        c.plot(x, np.maximum(sc[f"r_{pair}"], RATIO_FLOOR), color=PAIR_COLOURS[pair], marker="o", ms=2.5,
               lw=1.0, label=f"{pair} pair")
    c.set_yscale("log")
    c.set_ylim(bottom=RATIO_FLOOR)
    for ax in (a, b, c):
        ax.axvline(sc["current"], color="#999999", lw=0.8, ls="--")
        if PARAMS[sc["key"]].log:
            ax.set_xscale("log")
    _finish(a, xlabel, "Carrier shift from the free-atom line (Hz)", "Carrier shift")
    _finish(b, xlabel, "Trap frequency of |3,−3⟩ (kHz)", "Trap frequencies")
    _finish(c, xlabel, "Sideband / carrier, Δn = +1", "Sideband strength")
    b.legend(fontsize=7, framealpha=0.85)
    c.legend(fontsize=7, framealpha=0.85)
    _note(a, "dashed: current setting", "bottom-right")
    _note(c, f"ratios below {RATIO_FLOOR:.0e} drawn at {RATIO_FLOOR:.0e}", "bottom-right")


# ------------------------------------------------------------------- GUI
def run_interactive(mainloop=True):
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk

    sys.path.insert(0, str(HERE))
    try:
        import labplot
    except ImportError:
        labplot = None
    if labplot is None or not labplot.HAVE_MPL:
        raise SystemExit("labplot.py (beside sideband_explorer.py) and matplotlib are needed for the window")
    labplot.apply_style()

    class ParamRow:
        """Label, slider, typed value, unit and a one-line explanation."""

        def __init__(self, parent, param, on_change, small, wrap):
            self.p, self.on_change = param, on_change
            self.var = tk.DoubleVar(value=self._to_slider(param.default))
            self.text = tk.StringVar(value=param.fmt(param.default))
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=(5, 0))
            top = ttk.Frame(frame)
            top.pack(fill="x")
            ttk.Label(top, text=param.label).pack(side="left")
            ttk.Label(top, text=param.unit, width=6).pack(side="right")
            entry = ttk.Entry(top, textvariable=self.text, width=9, justify="right")
            entry.pack(side="right")
            entry.bind("<Return>", self._typed)
            entry.bind("<FocusOut>", self._typed)
            lo, hi = self._to_slider(param.lo), self._to_slider(param.hi)
            ttk.Scale(frame, from_=lo, to=hi, variable=self.var, command=self._slid).pack(fill="x")
            ttk.Label(frame, text=param.help, font=small, foreground="#666666", wraplength=wrap,
                      justify="left").pack(fill="x")

        def _to_slider(self, v):
            return math.log10(v) if self.p.log else v

        def get(self):
            v = self.var.get()
            return 10 ** v if self.p.log else v

        def set(self, v):
            self.var.set(self._to_slider(v))
            self.text.set(self.p.fmt(v))

        def _slid(self, _=None):
            self.text.set(self.p.fmt(self.get()))
            self.on_change(self.p.key)

        def _typed(self, _=None):
            try:
                v = float(self.text.get())
            except ValueError:
                self.text.set(self.p.fmt(self.get()))
                return
            v = min(max(v, self.p.lo), self.p.hi)
            if abs(self._to_slider(v) - self.var.get()) > 1e-12:
                self.set(v)
                self.on_change(self.p.key)
            else:
                self.text.set(self.p.fmt(v))

    class App:
        def __init__(self, root):
            self.root = root
            self.settings = default_settings()
            self.cache = {}
            self.ref = compute(default_settings(), self.cache)
            self.result = self.ref
            self.grads = None
            self.scan_result = None
            self.slope = zeeman_slope_hz_per_g(1.39304)
            self._gen = 0
            self._scan_cancel = False
            self._after = None
            self._loading = False
            self.jobs, self.results = queue.Queue(), queue.Queue()
            threading.Thread(target=self._work, daemon=True).start()

            root.title("Sideband explorer - lattice-sim")
            base = tkfont.nametofont("TkDefaultFont")
            small = base.copy()
            small.configure(size=max(7, base.cget("size") - 1))
            bold = base.copy()
            bold.configure(weight="bold")
            self.small = small

            outer = ttk.Panedwindow(root, orient="horizontal")
            outer.pack(fill="both", expand=True)
            left = self._scrollable(outer, width=370)
            right = ttk.Frame(outer)
            outer.add(left.outer, weight=0)
            outer.add(right, weight=1)

            # -- settings column
            body = left.inner
            wrap = 340
            intro = ("Microwave spectrum of |4,−4⟩ → |3,−3⟩ at one lattice site. A sideband appears when the "
                     "two states' wells differ: its position is a trap frequency, its height measures how far "
                     "the wells are apart. Type a value and press Enter, or drag.")
            ttk.Label(body, text=intro, wraplength=wrap, justify="left", font=small).pack(fill="x", pady=(2, 6))
            row = ttk.Frame(body)
            row.pack(fill="x")
            ttk.Label(row, text="Pair to edit:", font=bold).pack(side="left")
            self.pair_var = tk.StringVar(value="x")
            for ax in PAIRS:
                ttk.Radiobutton(row, text=f"{ax} pair", value=ax, variable=self.pair_var,
                                command=self._pair_chosen).pack(side="left", padx=4)
            ttk.Label(body, text="The next two groups belong to this pair; each pair keeps its own values.",
                      font=small, foreground="#666666", wraplength=wrap).pack(fill="x")

            self.rows = {}
            self.grp_align = ttk.LabelFrame(body, text="", padding=6)
            self.grp_align.pack(fill="x", pady=(8, 0))
            for p in PAIR_PARAMS[:4]:
                self.rows[p.key] = ParamRow(self.grp_align, p, self._changed, small, wrap - 20)
            ttk.Label(self.grp_align, text="Offset direction:").pack(anchor="w", pady=(6, 0))
            self.dir_var = tk.StringVar(value="pol")
            for value, text in (("pol", "along the static polarization"), ("fringe", "along the fringes")):
                ttk.Radiobutton(self.grp_align, text=text, value=value, variable=self.dir_var,
                                command=lambda: self._changed("offset_dir")).pack(anchor="w", padx=12)
            ttk.Label(self.grp_align, text="Only the gradient readout depends on it: the spectrum at the "
                      "central site depends on the distance alone.", font=small, foreground="#666666",
                      wraplength=wrap - 20, justify="left").pack(fill="x")

            self.grp_eom = ttk.LabelFrame(body, text="", padding=6)
            self.grp_eom.pack(fill="x", pady=(8, 0))
            for p in PAIR_PARAMS[4:]:
                self.rows[p.key] = ParamRow(self.grp_eom, p, self._changed, small, wrap - 20)
            hrow = ttk.Frame(self.grp_eom)
            hrow.pack(fill="x", pady=(6, 0))
            ttk.Label(hrow, text="Ellipticity handedness:").pack(side="left")
            self.hand_var = tk.IntVar(value=-1)
            for h in (-1, 1):
                ttk.Radiobutton(hrow, text=f"{h:+d}", value=h, variable=self.hand_var,
                                command=lambda: self._changed("handedness")).pack(side="left", padx=4)
            ttk.Label(self.grp_eom, text="Sense of the minor (quadrature) component. Not measured - the "
                      "polarimeter gives |ellipticity| only - so compare both.", font=small,
                      foreground="#666666", wraplength=wrap - 20, justify="left").pack(fill="x")

            grp = ttk.LabelFrame(body, text="Field and atoms (all pairs)", padding=6)
            grp.pack(fill="x", pady=(8, 0))
            for p in GLOBAL_PARAMS:
                self.rows[p.key] = ParamRow(grp, p, self._changed, small, wrap - 20)

            brow = ttk.Frame(body)
            brow.pack(fill="x", pady=8)
            ttk.Button(brow, text="Reset this pair", command=self._reset_pair).pack(side="left")
            ttk.Button(brow, text="Reset all", command=self._reset_all).pack(side="left", padx=6)

            # -- plots
            read = ttk.Frame(right, padding=(6, 4))
            read.pack(side="bottom", fill="x")
            nb = ttk.Notebook(right)
            nb.pack(side="top", fill="both", expand=True)
            self.book = labplot.PlotBook(nb)
            figs = str(REPO / "figures")
            self.p_spec = self.book.add("spectrum", "Spectrum", draw=self._draw_spectrum, figsize=(8, 4.6),
                                        save_name="sideband_spectrum", save_dir=figs)
            self.log_var = tk.BooleanVar(value=True)
            self.nom_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.p_spec.controls, text="log scale", variable=self.log_var,
                            command=self._redraw_spectrum).pack(side="left", padx=4)
            ttk.Checkbutton(self.p_spec.controls, text="show nominal settings", variable=self.nom_var,
                            command=self._redraw_spectrum).pack(side="left", padx=4)
            self.p_wells = self.book.add("wells", "Wells", draw=self._draw_wells, figsize=(8, 4.6),
                                         save_name=lambda: f"sideband_wells_{self.settings['edited']}",
                                         save_dir=figs)
            self.p_scan = self.book.add("scan", "Scan", draw=self._draw_scan, figsize=(8, 4.6),
                                        save_name="sideband_scan", save_dir=figs)
            c = self.p_scan.controls
            ttk.Label(c, text="Setting:").pack(side="left", padx=(4, 2))
            self.scan_names = {self._scan_name(p.key): p.key for p in PAIR_PARAMS + GLOBAL_PARAMS
                               if p.key not in DISPLAY_ONLY}
            self.scan_var = tk.StringVar(value=self._scan_name("theta_deg"))
            self.scan_box = ttk.Combobox(c, textvariable=self.scan_var, values=list(self.scan_names),
                                         state="readonly", width=42)
            self.scan_box.pack(side="left")
            self.scan_box.bind("<<ComboboxSelected>>", lambda e: self._scan_defaults())
            self.scan_lo, self.scan_hi, self.scan_n = tk.StringVar(), tk.StringVar(), tk.StringVar(value="15")
            for label, var, w in (("from", self.scan_lo, 8), ("to", self.scan_hi, 8), ("points", self.scan_n, 4)):
                ttk.Label(c, text=label).pack(side="left", padx=(6, 2))
                ttk.Entry(c, textvariable=var, width=w, justify="right").pack(side="left")
            self.scan_btn = ttk.Button(c, text="Run scan", command=self._run_scan)
            self.scan_btn.pack(side="left", padx=6)
            ttk.Button(c, text="Stop", command=self._stop_scan).pack(side="left")
            self.scan_status = ttk.Label(c, text="", foreground="#666666")
            self.scan_status.pack(side="left", padx=6)
            self._scan_defaults()

            # -- readout
            self.summary = ttk.Label(read, text="", justify="left")
            self.summary.pack(fill="x")
            cols = ("quantity", "x", "y", "z")
            self.tree = ttk.Treeview(read, columns=cols, show="headings", height=6)
            self.tree.heading("quantity", text="per pair: current (nominal)")
            self.tree.column("quantity", width=290, anchor="w")
            for ax in PAIRS:
                self.tree.heading(ax, text=f"{ax} pair")
                self.tree.column(ax, width=150, anchor="e")
            self.tree.pack(fill="x", pady=(4, 2))
            self.grad_label = ttk.Label(read, text="", justify="left", font=small)
            self.grad_label.pack(fill="x")
            self.status = ttk.Label(root, text="", foreground="#666666", anchor="w", padding=(6, 2))
            self.status.pack(fill="x", side="bottom")

            self._load_pair()
            self._show_result()
            root.protocol("WM_DELETE_WINDOW", self._close)
            root.after(60, self._poll)

        # .. layout helpers
        def _scrollable(self, parent, width):
            outer = ttk.Frame(parent)
            canvas = tk.Canvas(outer, width=width, highlightthickness=0)
            bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas, padding=(8, 4))
            inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            win = canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
            canvas.configure(yscrollcommand=bar.set)
            canvas.pack(side="left", fill="both", expand=True)
            bar.pack(side="right", fill="y")

            def wheel(event):
                under = self.root.winfo_containing(event.x_root, event.y_root)
                if under is not None and str(under).startswith(str(outer)):
                    canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")
            self.root.bind_all("<MouseWheel>", wheel, add="+")
            return type("Scroll", (), dict(outer=outer, inner=inner))

        # .. settings <-> widgets
        def _load_pair(self):
            ax = self.settings["edited"]
            self._loading = True
            for p in PAIR_PARAMS:
                self.rows[p.key].set(self.settings["pairs"][ax][p.key])
            self.hand_var.set(self.settings["pairs"][ax]["handedness"])
            for p in GLOBAL_PARAMS:
                self.rows[p.key].set(self.settings["glob"][p.key])
            self.dir_var.set(self.settings["offset_dir"])
            self._loading = False
            self.grp_align.configure(text=f"Beam alignment - {ax} pair")
            self.grp_eom.configure(text=f"EOM arm polarization - {ax} pair")
            self._scan_box_refresh()

        def _read_widgets(self):
            ax = self.settings["edited"]
            for p in PAIR_PARAMS:
                self.settings["pairs"][ax][p.key] = float(self.rows[p.key].get())
            self.settings["pairs"][ax]["handedness"] = int(self.hand_var.get())
            for p in GLOBAL_PARAMS:
                self.settings["glob"][p.key] = float(self.rows[p.key].get())
            self.settings["offset_dir"] = self.dir_var.get()

        def _changed(self, key):
            if self._loading:
                return
            self._read_widgets()
            if key in DISPLAY_ONLY:
                self._redraw_spectrum()
                return
            if self._after is not None:
                self.root.after_cancel(self._after)
            self._after = self.root.after(120, self._submit)
            self.status.configure(text="computing ...")

        def _pair_chosen(self):
            self._read_widgets()
            self.settings["edited"] = self.pair_var.get()
            self._load_pair()
            self._submit()

        def _reset_pair(self):
            ax = self.settings["edited"]
            self.settings["pairs"][ax] = default_settings()["pairs"][ax]
            self._load_pair()
            self._submit()

        def _reset_all(self):
            ax = self.settings["edited"]
            self.settings = default_settings()
            self.settings["edited"] = ax
            self._load_pair()
            self._submit()

        # .. worker
        def _submit(self):
            self._after = None
            self._gen += 1
            self.jobs.put(("update", self._gen, copy_settings(self.settings)))
            self.status.configure(text="computing ...")

        def _work(self):
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                kind, gen, payload = job
                try:
                    if kind == "update":
                        if gen != self._gen:
                            continue                 # superseded while waiting
                        self.results.put(("spectrum", gen, compute(payload, self.cache)))
                        if needs_gradients(payload) and gen == self._gen:
                            t0 = time.perf_counter()
                            g = compute_gradients(payload, self.cache)
                            self.results.put(("gradients", gen, (g, time.perf_counter() - t0)))
                    elif kind == "scan":
                        key, values = payload["key"], payload["values"]
                        sc = scan(payload["settings"], key, values, self.cache,
                                  progress=lambda i, n: self.results.put(("scan_progress", gen, (i, n))),
                                  cancelled=lambda: self._scan_cancel)
                        self.results.put(("scan", gen, sc))
                except Exception:
                    self.results.put(("error", gen, traceback.format_exc()))

        def _poll(self):
            try:
                while True:
                    kind, gen, payload = self.results.get_nowait()
                    if kind == "spectrum" and gen == self._gen:
                        self.result = payload
                        self.grads = None
                        self._show_result()
                    elif kind == "gradients" and gen == self._gen:
                        self.grads = payload
                        self._show_gradients()
                    elif kind == "scan_progress":
                        self.scan_status.configure(text=f"{payload[0]} / {payload[1]}")
                    elif kind == "scan":
                        self.scan_btn.state(["!disabled"])
                        if payload is None:
                            self.scan_status.configure(text="stopped")
                        else:
                            self.scan_result = payload
                            self.scan_status.configure(text=f"{payload['values'].size} points")
                            self.p_scan.mark_dirty()
                            self.p_scan.draw_if_visible()
                    elif kind == "error":
                        self.status.configure(text="calculation failed - see the console", foreground="#c60")
                        print(payload, file=sys.stderr)
            except queue.Empty:
                pass
            self.root.after(60, self._poll)

        def _close(self):
            self._scan_cancel = True
            self.jobs.put(None)
            self.root.destroy()

        # .. scan
        def _scan_name(self, key):
            p = PARAMS[key]
            return (p.label if key in self.settings["glob"] else f"{self.settings['edited']} pair: {p.label}") \
                + (f" ({p.unit})" if p.unit else "")

        def _scan_box_refresh(self):
            key = self.scan_names.get(self.scan_var.get(), "theta_deg")
            self.scan_names = {self._scan_name(p.key): p.key for p in PAIR_PARAMS + GLOBAL_PARAMS
                               if p.key not in DISPLAY_ONLY}
            self.scan_box.configure(values=list(self.scan_names))
            self.scan_var.set(self._scan_name(key))

        def _scan_defaults(self):
            p = PARAMS[self.scan_names[self.scan_var.get()]]
            self.scan_lo.set(p.fmt(p.lo))
            self.scan_hi.set(p.fmt(p.hi))

        def _run_scan(self):
            self._read_widgets()
            key = self.scan_names[self.scan_var.get()]
            p = PARAMS[key]
            try:
                lo, hi, n = float(self.scan_lo.get()), float(self.scan_hi.get()), int(self.scan_n.get())
            except ValueError:
                self.scan_status.configure(text="from, to and points must be numbers")
                return
            n = min(max(n, 2), 101)
            values = np.geomspace(lo, hi, n) if p.log and lo > 0 and hi > 0 else np.linspace(lo, hi, n)
            self._scan_cancel = False
            self.scan_btn.state(["disabled"])
            self.scan_status.configure(text=f"0 / {n}")
            self.jobs.put(("scan", 0, dict(settings=copy_settings(self.settings), key=key, values=values)))

        def _stop_scan(self):
            self._scan_cancel = True

        # .. display
        def _draw_spectrum(self, fig):
            draw_spectrum(fig, self.result, self.ref, self.settings["glob"]["width_hz"],
                          log=self.log_var.get(), show_nominal=self.nom_var.get())

        def _draw_wells(self, fig):
            draw_wells(fig, self.result, self.result["settings"]["edited"])

        def _draw_scan(self, fig):
            draw_scan(fig, self.scan_result)

        def _redraw_spectrum(self):
            self.p_spec.mark_dirty()
            self.p_spec.draw_if_visible()

        def _show_result(self):
            res, ref = self.result, self.ref
            sp, rsp = res["sp"], ref["sp"]
            dz = sp["zeeman_hz"] - rsp["zeeman_hz"]
            self.summary.configure(text=(
                f"Carrier: {sp['carrier_shift_hz']:+.1f} Hz from the free-atom line "
                f"(nominal {rsp['carrier_shift_hz']:+.1f} Hz).    "
                f"Free-atom line: {dz / 1e3:+.3f} kHz from its value at 1.39304 G "
                f"({self.slope / 1e6:+.3f} kHz per mG)."))
            self.tree.delete(*self.tree.get_children())
            rows = [
                ("trap frequency of |3,−3⟩ (kHz)", lambda s, ax: s["axes"][ax]["f_lower_hz"] / 1e3, "{:.2f}"),
                ("trap frequency of |4,−4⟩ (kHz)", lambda s, ax: s["axes"][ax]["f_upper_hz"] / 1e3, "{:.2f}"),
                ("well separation x(|3,−3⟩) - x(|4,−4⟩) (nm)",
                 lambda s, ax: s["axes"][ax]["displacement_m"] * 1e9, "{:+.1f}"),
                ("sideband / carrier, Δn = +1", lambda s, ax: sideband_ratios(s, ax)[0], "ratio"),
                ("sideband / carrier, Δn = −1", lambda s, ax: sideband_ratios(s, ax)[1], "ratio"),
                ("mean vibrational occupation", mean_occupation, "{:.2f}"),
            ]
            def show(v, fmt):
                if fmt == "ratio":
                    return f"{v:.3f}" if v >= 1e-3 else f"{v:.1e}"
                return fmt.format(v)
            for name, fn, fmt in rows:
                self.tree.insert("", "end", values=[name] + [
                    f"{show(fn(sp, ax), fmt)}  ({show(fn(rsp, ax), fmt)})" for ax in PAIRS])
            self._show_gradients()
            took = "from cache" if res["seconds"] < 0.01 else f"computed in {res['seconds']:.2f} s"
            self.status.configure(text=f"{took}; {res['n_levels']} vibrational levels per pair",
                                  foreground="#666666")
            self.book.mark_dirty(["spectrum", "wells"])
            self.book.draw_visible()

        def _show_gradients(self):
            s = self.result["settings"]
            ax = s["edited"]
            if not needs_gradients(s):
                self.grad_label.configure(text=f"Gradients across the lattice: shown when an arm of the {ax} "
                                               "pair is off-centre.")
                return
            if self.grads is None:
                self.grad_label.configure(text="Gradients across the lattice: computing ...")
                return
            g, secs = self.grads
            along = "the fringes" if s["offset_dir"] == "fringe" else "the static polarization"
            self.grad_label.configure(text=(
                f"Gradients across the lattice (sites ±{GRADIENT_HALF_SPAN_M * 1e6:.0f} µm from the centre) "
                f"from the {ax}-pair arm offsets along {along}:\n"
                f"   carrier        {g['carrier_hz_per_m_fringe']:+8.2f} Hz/µm along the fringes, "
                f"{g['carrier_hz_per_m_pol']:+8.2f} Hz/µm along the polarization\n"
                f"   f(|3,−3⟩)      {g['f_lower_hz_per_m_fringe']:+8.2f} Hz/µm along the fringes, "
                f"{g['f_lower_hz_per_m_pol']:+8.2f} Hz/µm along the polarization\n"
                f"   for scale: a 0.1 G/cm gradient of |B| moves the carrier by {self.slope * 1e-5:+.1f} Hz/µm"
                f"   ({secs:.1f} s)"))

    root = tk.Tk()
    root.geometry("1400x900")
    app = App(root)
    if mainloop:
        root.mainloop()
    return app


def save_nominal(path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    res = compute(default_settings())
    fig = Figure(figsize=(8, 4.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    draw_spectrum(fig, res, res, 300.0, show_nominal=False)
    fig.savefig(path, dpi=150)
    sp = res["sp"]
    print(f"carrier shift {sp['carrier_shift_hz']:+.1f} Hz")
    for ax in PAIRS:
        a = sp["axes"][ax]
        r_plus, r_minus = sideband_ratios(sp, ax)
        print(f"{ax}: f(3,-3) {a['f_lower_hz'] / 1e3:6.2f} kHz  f(4,-4) {a['f_upper_hz'] / 1e3:6.2f} kHz  "
              f"separation {a['displacement_m'] * 1e9:+6.1f} nm  sideband/carrier +1 {r_plus:.3f}  -1 {r_minus:.3f}")


if __name__ == "__main__":
    import os as _os
    _os.chdir(REPO)
    if "--save" in sys.argv:
        save_nominal(sys.argv[sys.argv.index("--save") + 1])
    else:
        run_interactive()
