"""Sideband-explorer extension: the spectrum across a 9 x 9 x 9 block of
lattice sites (``lattice_sim.spectra.grid``).

The lab looks at the array in two ways (Data-Analysis/readstats9p0
2026PC.py), and the three tabs follow them:

* Lattice grid - one number per site as a map of each plane, "Z = k" maps
  of the y x x sites in a 3 x 3 panel layout (readstats plottemp /
  analyzeSiteMap), here for any plane orientation, with a 3D plane fit of
  the gradient per site (analyzeSiteMap's planefit3d);
* Lattice slices - the alignment slices (readstats plotslices): the
  spectrum summed over the other two axes, one curve per column x, row y
  and plane z, stacked 0.5 apart, and the carrier position and a pair's
  sideband spacing followed across the slices: a straight line through the
  carrier (its slope is the alignment gradient per site) and a parabola
  through the spacing (its vertex is the slice the beams are centred on);
* Array spectrum - the spectrum of the whole block (what a scan of the whole
  array records) against the centre site, and line cuts of the chosen
  quantity through the centre along x, y and z.

Each pair's arms are offset as in the settings column (each pair's own
values, along the chosen offset direction), and each arm's field at a site
follows from its distance to the arm's axis with the beams' Gaussian
envelope (w = WAIST_M of the explorer). The block is computed in short
steps (CHUNK_S) on the explorer's worker thread, so the window stays
responsive and the explorer's own spectra run in between: 75 level
calculations (25 per pair, ~5 s) and 729 site spectra (~2-3 s) the first
time; the level calculations are cached, so a change to one pair redoes only
its 25, and the last STORE_BLOCKS blocks are kept, so returning to settings
already seen (the nominal ones, say) is immediate. "change from the same
site with every arm centred" computes that second block when chosen.
Quantities other extensions register in app.grid_quantities are evaluated
on the worker thread too, QUANTITY_BLOCK sites per step (their functions
must not touch Tk), and re-evaluated when app.quantity_changed(name) is
announced.
"""

import sys
import time
import traceback
from dataclasses import replace

import numpy as np

from lattice_sim.spectra import grid as G

CHUNK_S = 0.4                    # worker time per step before the window gets a turn
QUANTITY_BLOCK = 81              # sites per worker step when evaluating another extension's quantity
STORE_BLOCKS = 4                 # computed blocks kept (about 15 MB each): returning to them is instant
SLICE_OFFSET = 0.5               # readstats plotslices: vertical spacing per slice
SLICE_GUIDE_KHZ = 4.0            # readstats slices_v_line_separation (kHz)
PLANES = {"x–y planes (imaging plane), one per z": 2, "x–z planes, one per y": 1, "y–z planes, one per x": 0}
REFERENCES = ("value at each site", "change from the centre site",
              "change from the same site with every arm centred")
FOLLOW = ("carrier", "x-pair sideband spacing", "y-pair sideband spacing", "z-pair sideband spacing")
AXES = "xyz"
VALUE_CMAP = "cividis"
DEVIATION_CMAP = "RdBu_r"


# ------------------------------------------------------------------ model
def setups_from_settings(se, settings, centred=False):
    """(SiteConfig of the centre site, {axis: PairSetup}) for the explorer settings;
    ``centred`` puts every arm on the atoms (same depth, angle, ER, crossing)."""
    site = se.site_from(settings)
    out = {}
    for cfg in site.axes:
        s = settings["pairs"][cfg.axis]
        # site_from scales the trap frequency by 1 / spacing; PairSetup.base is at the nominal spacing
        base = replace(cfg, trap_frequency_hz=cfg.trap_frequency_hz * cfg.spacing_m / G.NOMINAL_SPACING_M)
        so, ro = (0.0, 0.0) if centred else (s["static_um"] * 1e-6, s["rotating_um"] * 1e-6)
        dalpha = np.deg2rad(s["dalpha_deg"])
        if settings["offset_dir"] == "fringe":
            # the explorer's offset is the arm's distance from the atoms; an arm at +/-alpha to the
            # bisector whose axis is moved by r along the fringes passes the atoms at r cos(alpha)
            cos_a = np.cos(G.NOMINAL_HALF_ANGLE + dalpha)
            so, ro = so / cos_a, ro / cos_a
        out[cfg.axis] = G.pair_setup(base, so, ro, settings["offset_dir"], dalpha, se.WAIST_M)
    return site, out


def any_offset(settings):
    return any(p["static_um"] != 0 or p["rotating_um"] != 0 for p in settings["pairs"].values())


def centred_settings(se, settings):
    """The same settings with every arm on the atoms (the reference block)."""
    s = se.copy_settings(settings)
    for p in s["pairs"].values():
        p["static_um"] = p["rotating_um"] = 0.0
    return s


def grid_settings_key(se, settings):
    """What the grid depends on (the display linewidth and the edited pair do not;
    the offset direction only when an arm is off-centre)."""
    pairs = tuple((ax, tuple(sorted(v.items()))) for ax, v in sorted(settings["pairs"].items()))
    glob = tuple(sorted((k, v) for k, v in settings["glob"].items() if k not in se.DISPLAY_ONLY))
    return pairs, glob, settings["offset_dir"] if any_offset(settings) else None


class GridWork:
    """One grid calculation in steps; touched only by the worker thread."""

    def __init__(self, se, settings, cache):
        self.se, self.settings, self.cache = se, settings, cache
        self.job = None
        self.t0 = time.perf_counter()

    def step(self, cancelled):
        if self.job is None:
            site, setups = setups_from_settings(self.se, self.settings)
            self.job = G.GridJob(site, setups, self.cache)
        if not self.job.step(CHUNK_S, cancelled):
            return ("running",) + self.job.progress
        return ("done", self.job.finish(), time.perf_counter() - self.t0)


# ------------------------------------------------------------- drawing
def _fmt(v):
    if not np.isfinite(v):
        return "nan"
    a = abs(v)
    return f"{v:.3g}" if (a >= 1e4 or a < 1e-2) and a != 0 else f"{v:.4g}"


def displayed(values, ref_values, reference):
    """The map values for the reference choice (REFERENCES)."""
    if reference == REFERENCES[1]:
        h = values.shape[0] // 2
        return values - values[h, h, h]
    if reference == REFERENCES[2]:
        return values - ref_values
    return values


def colour_label(name, reference):
    if reference == REFERENCES[1]:
        return f"{name}\nminus the centre site"
    if reference == REFERENCES[2]:
        return f"{name}\nminus the same site, arms centred"
    return name


def _norm_limits(arrays, reference):
    finite = np.concatenate([np.ravel(a)[np.isfinite(np.ravel(a))] for a in arrays])
    if finite.size == 0:
        return 0.0, 1.0
    if reference == REFERENCES[0]:
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo <= 1e-12 * max(1.0, abs(hi)):
            pad = max(abs(hi) * 1e-6, 1e-9)
            return lo - pad, hi + pad
        return lo, hi
    m = float(np.max(np.abs(finite)))
    m = m if m > 0 else 1.0
    return -m, m


def plane_of(values, normal, s, half):
    """2D slice (first remaining axis, second remaining axis) at index s along ``normal``."""
    return np.take(values, s + half, axis=normal)


def axis_names(normal):
    return [AXES[a] for a in range(3) if a != normal]


def settings_note(se, settings, column=False):
    """The alignment settings that make the sites differ (one line, or a narrow column)."""
    lines = []
    for ax in AXES:
        p = settings["pairs"][ax]
        if p["static_um"] or p["rotating_um"] or p["dalpha_deg"]:
            parts = [f"static arm {p['static_um']:+.1f} µm", f"EOM arm {p['rotating_um']:+.1f} µm",
                     f"crossing half-angle change {p['dalpha_deg']:+.3f} deg"]
            lines.append(f"{ax} pair:\n   " + "\n   ".join(parts) if column else f"{ax} pair: " + ", ".join(parts))
    along = "the static polarization" if settings["offset_dir"] == "pol" else "the fringes"
    tail = [f"offsets along {along}" if lines else "all arms centred", f"waist {se.WAIST_M * 1e6:.0f} µm"]
    lines.extend(tail if column else ["; ".join(tail)])
    return "\n".join(lines) if column else "; ".join(lines)


ZERO_FRACTION = 1e-9             # fitted gradients below this fraction of the data range print as 0


def gradient_note(fit, spacing, values=None, multiline=False):
    """The plane-fit gradients per site and per µm (values only)."""
    g = np.array(fit["gradient"], float)
    if values is not None:
        span = float(np.nanmax(values) - np.nanmin(values)) if np.isfinite(values).any() else 0.0
        g[np.abs(g * max(spacing.values())) <= ZERO_FRACTION * max(span, 1e-300)] = 0.0
    parts = [f"{AXES[a]}: {_fmt(g[a] * spacing[AXES[a]])} per site ({_fmt(g[a] * 1e-6)} per µm)" for a in range(3)]
    sep = "\n" if multiline else "   "
    return "linear fit over all sites:" + ("\n" if multiline else " ") + sep.join(parts)


def _side_text(sub, text):
    sub.text(0.02, 0.97, text, ha="left", va="top", fontsize=7, color="#555555", transform=sub.transSubfigure)


def map_info(se, result, values, settings, extra=""):
    sp = ", ".join(f"{a} {result.spacing_m[a] * 1e6:.3f}" for a in AXES)
    return (f"{result.index.size}³ sites, periods (µm):\n{sp}\n\n" + settings_note(se, settings, column=True) + "\n\n"
            + gradient_note(G.fit_plane(values, result.positions_m), result.spacing_m, values, multiline=True)
            + "\n(of the values, before any subtraction)" + extra
            + f"\n\nT = {result.temperature_k * 1e6:.2f} µK, {result.n_levels} levels per pair")


def draw_maps(fig, se, result, ref, values, ref_values, name, reference, normal, s, all_planes, settings):
    """The Lattice grid tab: one plane, or all planes of one orientation (3 x 3),
    with the settings and the plane fit in a text column on the right."""
    fig.clear()
    half = result.index.size // 2
    shown = displayed(values, ref_values, reference)
    cmap = VALUE_CMAP if reference == REFERENCES[0] else DEVIATION_CMAP
    h_name, v_name = axis_names(normal)
    n_name = AXES[normal]
    a_n = result.spacing_m[n_name] * 1e6
    extent = [-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]
    sub_map, sub_txt = fig.subfigures(1, 2, width_ratios=[3.3, 1])
    _side_text(sub_txt, map_info(se, result, values, settings))
    if all_planes:
        planes = list(range(-half, half + 1))
        vmin, vmax = _norm_limits([shown], reference)
        cols = 3
        rows = int(np.ceil(len(planes) / cols))
        axs = np.atleast_1d(sub_map.subplots(rows, cols)).ravel()
        im = None
        for ax, k in zip(axs, planes):
            im = ax.imshow(plane_of(shown, normal, k, half).T, origin="lower", extent=extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{n_name} = {k} ({k * a_n:+.1f} µm)", fontsize=7)
            ax.set_xticks([-half, 0, half])
            ax.set_yticks([-half, 0, half])
            ax.tick_params(labelsize=6)
            ax.set_xlabel(f"{h_name} site", fontsize=6, labelpad=1)
            ax.set_ylabel(f"{v_name} site", fontsize=6, labelpad=1)
        for ax in axs[len(planes):]:
            ax.set_visible(False)
        cb = sub_map.colorbar(im, ax=axs.tolist(), shrink=0.9)
        cb.set_label(colour_label(name, reference), fontsize=7)
        cb.ax.tick_params(labelsize=6)
        sub_map.suptitle(f"{name}: every {h_name}–{v_name} plane of the block (one colour scale)", fontsize=8)
        return
    ax = sub_map.add_subplot(111)
    arr = plane_of(shown, normal, s, half)
    vmin, vmax = _norm_limits([arr], reference)
    im = ax.imshow(arr.T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    a_h, a_v = (result.spacing_m[c] * 1e6 for c in (h_name, v_name))
    ax.set_xticks(result.index)
    ax.set_yticks(result.index)
    ax.set_xlabel(f"Site index along {h_name} (period {a_h:.3f} µm)")
    ax.set_ylabel(f"Site index along {v_name} (period {a_v:.3f} µm)")
    ax.set_title(f"{name}\n{h_name}–{v_name} plane {n_name} = {s} ({s * a_n:+.2f} µm)", fontsize=8)
    cb = sub_map.colorbar(im, ax=ax)
    cb.set_label(colour_label(name, reference), fontsize=7)


def _room(ax, frac=0.35):
    """Extend the y range upwards so a note at the top does not cover data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def slice_follow(result, follow):
    """Per-slice (3, n) values followed across the slices and their unit:
    the carrier centroid (Hz from the free-atom line, carrier-weighted mean
    over the slice's sites) or a pair's Δn = +1 spacing f(|3,−3⟩) (kHz,
    weighted by that sideband's strength)."""
    if follow == FOLLOW[0]:
        v = result.quantity(lambda sp: sp["carrier_shift_hz"])
        w = result.quantity(lambda sp: sp["carrier_weight"])
        unit, scale = "Hz", 1.0
    else:
        ax = follow[0]
        v = result.quantity(lambda sp: sp["axes"][ax]["f_lower_hz"])
        w = result.quantity(lambda sp: G.collapse_lines(sp["lines"], ax).get((ax, 1), (0.0, 0.0))[1])
        unit, scale = "kHz", 1e-3
    out = []
    for a in range(3):
        others = tuple(b for b in range(3) if b != a)
        with np.errstate(invalid="ignore", divide="ignore"):     # all weights 0 at mid angles: NaN
            out.append(np.nansum(v * w, axis=others) / np.nansum(w, axis=others) * scale)
    return np.array(out), unit


def detuning_axis(result, centre_sp):
    f_max = 1.25 * max(a[k] for a in centre_sp["axes"].values() for k in ("f_upper_hz", "f_lower_hz"))
    return np.linspace(-f_max, f_max, 4001)


def draw_slices(fig, se, result, width_hz, follow, settings):
    """The Lattice slices tab, after readstats plotslices."""
    fig.clear()
    axs = fig.subplots(2, 3, gridspec_kw=dict(height_ratios=[2.2, 1]))
    centre = result.centre()
    f = detuning_axis(result, centre)
    n = result.index.size
    curves = [G.slice_curves(result, a, f, width_hz) for a in range(3)]
    top = max(float(c.max()) for c in curves)
    titles = {0: "Slices along x (each summed over y and z)", 1: "Slices along y (each summed over x and z)",
              2: "Slices along z (each summed over x and y)"}
    guides = np.arange(np.ceil(f[0] / 1e3 / SLICE_GUIDE_KHZ), np.floor(f[-1] / 1e3 / SLICE_GUIDE_KHZ) + 1) \
        * SLICE_GUIDE_KHZ
    for a in range(3):
        ax = axs[0, a]
        for s in range(n):
            ax.plot(f / 1e3, curves[a][s] / top + SLICE_OFFSET * s, color=se.CURRENT_COLOUR, lw=0.8)
        ax.vlines(guides, 0, SLICE_OFFSET * (n - 1) + 1, linestyles="dotted", colors="#aaaaaa", lw=0.6)
        ax.set_ylim(0, SLICE_OFFSET * (n - 1) + 1.05)
        ax.set_xlim(f[0] / 1e3, f[-1] / 1e3)
        ax.set_yticks([SLICE_OFFSET * s for s in range(n)])
        ax.set_yticklabels([f"{AXES[a]} = {i}" for i in result.index], fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("Detuning from the free-atom line (kHz)", fontsize=7)
        ax.set_ylabel(f"Line strength, stacked {SLICE_OFFSET} per slice", fontsize=7)
        ax.set_title(titles[a], fontsize=7)
    values, unit = slice_follow(result, follow)
    for a in range(3):
        ax = axs[1, a]
        y = values[a]
        idx = result.index.astype(float)
        a_um = result.spacing_m[AXES[a]] * 1e6
        ax.plot(idx, y, "o", ms=3, color="#222222", label="per slice")
        xx = np.linspace(idx[0], idx[-1], 50)
        span = float(np.nanmax(y) - np.nanmin(y))
        slope, icpt = G.line_fit(idx, y)
        if abs(slope) * idx.size <= ZERO_FRACTION * max(span, abs(float(np.nanmean(y))), 1e-300):
            slope = 0.0                                          # round-off of a symmetric profile
        if follow == FOLLOW[0]:
            ax.plot(xx, slope * xx + icpt, color=se.CURRENT_COLOUR, lw=1.0, label="straight line")
            text = f"slope {_fmt(slope)} Hz/slice ({_fmt(slope / a_um)} Hz/µm)"
        else:
            vertex, curv = G.parabola_vertex(idx, y)
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                c2, c1, c0 = np.polyfit(idx[ok], y[ok], 2)
                ax.plot(xx, c2 * xx**2 + c1 * xx + c0, color=se.CURRENT_COLOUR, lw=1.0, label="parabola")
            text = (f"slope {_fmt(slope * 1e3)} Hz/slice ({_fmt(slope * 1e3 / a_um)} Hz/µm)\n"
                    f"vertex at slice {round(vertex, 2) + 0.0:+.2f}, curvature {_fmt(curv * 1e3)} Hz/slice²")
        ax.set_xticks(result.index)
        ax.tick_params(labelsize=6)
        lab = "Carrier (Hz)" if follow == FOLLOW[0] else f"{follow[0]}-pair spacing f(|3,−3⟩) (kHz)"
        se._finish(ax, f"Slice index along {AXES[a]} (period {a_um:.3f} µm)", lab)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        _room(ax, 0.6)
        se._note(ax, text, "top-left")
    fig.suptitle("Microwave spectrum |4,−4⟩ → |3,−3⟩ per slice of the lattice block; "
                 + settings_note(se, settings)
                 + f"\nline half-width {width_hz:.0f} Hz (display only); dotted lines every "
                 f"{SLICE_GUIDE_KHZ:.0f} kHz; curves scaled by one common factor", fontsize=7)


def draw_array(fig, se, result, ref, width_hz, values, ref_values, name, reference, settings, log=True):
    """The Array spectrum tab: whole block vs centre site, and line cuts."""
    fig.clear()
    a, b = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.5, 1]))
    centre = result.centre()
    f = detuning_axis(result, centre)
    ens = G.ensemble_curve(result.spectra, f, width_hz)
    one = se.spectrum_curve(centre, f, width_hz)
    a.plot(f / 1e3, one, color="#222222", lw=1.0, label="centre site")
    a.plot(f / 1e3, ens, color=se.CURRENT_COLOUR, lw=1.2, label=f"mean of all {len(result.spectra)} sites")
    if ref is not None and ref is not result:
        a.plot(f / 1e3, G.ensemble_curve(ref.spectra, f, width_hz), color=se.NOMINAL_COLOUR, lw=1.0, ls="--",
               label="all sites, every arm centred")
    if log:
        a.set_yscale("log")
        a.set_ylim(1e-5, 3.0)
    a.set_xlim(f[0] / 1e3, f[-1] / 1e3)
    se._finish(a, "Detuning from the free-atom |4,−4⟩ → |3,−3⟩ line (kHz)",
               "Line strength (undisplaced carrier at T = 0 = 1)",
               "Spectrum of the whole block and of the centre site")
    a.legend(loc="upper right", fontsize=7, framealpha=0.85)
    c = np.array([sp["carrier_shift_hz"] for sp in result.spectra if sp is not None])
    wts = np.array([sp["carrier_weight"] for sp in result.spectra if sp is not None])
    total = float(np.sum(wts))
    mean = float(np.sum(c * wts) / total) if total > 0 else float("nan")       # carrier weight 0 at mid angles
    rms = float(np.sqrt(np.sum(wts * (c - mean) ** 2) / total)) if total > 0 else float("nan")
    se._note(a, f"carrier over the sites: weighted mean {mean:+.1f} Hz, rms spread {rms:.1f} Hz\n"
                f"centre site {centre['carrier_shift_hz']:+.1f} Hz; line half-width {width_hz:.0f} Hz (display only)",
             "top-left")
    shown = displayed(values, ref_values, reference)
    h = result.index.size // 2
    cuts = {"x": shown[:, h, h], "y": shown[h, :, h], "z": shown[h, h, :]}
    for ax_name, y in cuts.items():
        pos = result.index * result.spacing_m[ax_name] * 1e6
        b.plot(pos, y, marker="o", ms=3, lw=1.0, color=se.PAIR_COLOURS[ax_name], label=f"along {ax_name}")
    se._finish(b, "Site position from the centre (µm)", colour_label(name, reference),
               "Line cuts through the centre site")
    b.yaxis.label.set_size(7)
    _room(b, 0.6)
    b.legend(fontsize=7, framealpha=0.85, loc="upper left")
    se._note(b, gradient_note(G.fit_plane(values, result.positions_m), result.spacing_m, values, multiline=True)
             + "\n(of the values, before any subtraction)", "top-right")


# ------------------------------------------------------------------ GUI
class GridView:
    def __init__(self, app):
        self.app, self.se = app, app.se
        tk, ttk = app.tk, app.ttk
        self.cache = G.NodeCache()       # node level calculations (worker thread only)
        self.store = {}                  # settings key -> GridResult, the last STORE_BLOCKS blocks
        self.settings = None
        self.result = self.ref = None
        self.result_key = None
        self.run_id = 0
        self.handle = None
        self.running = False
        self.listener_calls = 0
        self._qcache = {}
        self._q_pending = None
        self.builtins = G.builtin_quantities()

        grp = app.add_settings_group(f"Lattice grid: {2 * G.GRID_HALF + 1}³ sites")
        app.help_label(grp, (
            f"The spectrum at every site of a {2 * G.GRID_HALF + 1} × {2 * G.GRID_HALF + 1} × "
            f"{2 * G.GRID_HALF + 1} block (indices −{G.GRID_HALF} … {G.GRID_HALF} along x, y, z; site period = "
            "each pair's lattice period, 4.84 µm nominal). Each arm's field at a site follows from its distance "
            f"to the arm's axis (Gaussian beam, w = {self.se.WAIST_M * 1e6:.0f} µm), with the arm offsets and "
            "crossing angles of all three pairs set above, along the chosen offset direction. Shown in the "
            "Lattice grid, Lattice slices and Array spectrum tabs."))
        row = ttk.Frame(grp)
        row.pack(fill="x", pady=(4, 0))
        self.btn = ttk.Button(row, text="Compute grid", command=self.start)
        self.btn.pack(side="left")
        ttk.Button(row, text="Stop", command=self.stop).pack(side="left", padx=6)
        self.status = ttk.Label(row, text="not computed", foreground="#666666")
        self.status.pack(side="left", padx=4)
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="recompute when the settings change (while a grid tab is shown)",
                        variable=self.auto_var).pack(anchor="w", pady=(4, 0))
        self.timing = app.help_label(grp, "")

        # -- tabs
        self.p_map = app.add_tab("grid", "Lattice grid", self._draw_map, save_name=self._save_map)
        self.p_slices = app.add_tab("slices", "Lattice slices", self._draw_slices, save_name="lattice_slices")
        self.p_array = app.add_tab("array", "Array spectrum", self._draw_array, save_name="array_spectrum")
        c = self.p_map.controls
        ttk.Label(c, text="Quantity:").pack(side="left", padx=(4, 2))
        self.q_var = tk.StringVar(value=next(iter(self.builtins)))
        self.q_box = ttk.Combobox(c, textvariable=self.q_var, state="readonly", width=44,
                                  values=self.quantity_names(), postcommand=self._refresh_names)
        self.q_box.pack(side="left")
        self.q_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(c, text=" show:").pack(side="left")
        self.ref_var = tk.StringVar(value=REFERENCES[1])
        box = ttk.Combobox(c, textvariable=self.ref_var, state="readonly", width=46, values=REFERENCES)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._reference_changed())
        c2 = ttk.Frame(self.p_map)
        c2.pack(side="top", fill="x", after=self.p_map.controls)
        ttk.Label(c2, text="Planes:").pack(side="left", padx=(4, 2))
        self.plane_var = tk.StringVar(value=next(iter(PLANES)))
        box = ttk.Combobox(c2, textvariable=self.plane_var, state="readonly", width=34, values=list(PLANES))
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self.redraw(["grid"]))
        ttk.Label(c2, text=" plane index:").pack(side="left")
        self.s_var = tk.IntVar(value=0)
        spin = ttk.Spinbox(c2, from_=-G.GRID_HALF, to=G.GRID_HALF, textvariable=self.s_var, width=4,
                           command=lambda: self.redraw(["grid"]))
        spin.pack(side="left")
        spin.bind("<Return>", lambda e: self.redraw(["grid"]))
        self.all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(c2, text="all planes (3 × 3, one colour scale)", variable=self.all_var,
                        command=lambda: self.redraw(["grid"])).pack(side="left", padx=8)
        c = self.p_slices.controls
        ttk.Label(c, text="Follow across the slices:").pack(side="left", padx=(4, 2))
        self.follow_var = tk.StringVar(value=FOLLOW[0])
        box = ttk.Combobox(c, textvariable=self.follow_var, state="readonly", width=26, values=FOLLOW)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self.redraw(["slices"]))
        c = self.p_array.controls
        self.log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(c, text="log scale", variable=self.log_var,
                        command=lambda: self.redraw(["array"])).pack(side="left", padx=4)
        ttk.Label(c, text="  line cuts: the quantity and display chosen in the Lattice grid tab",
                  foreground="#666666").pack(side="left")
        for p in (self.p_map, self.p_slices, self.p_array):
            p.show_placeholder("Press Compute grid (settings column) to compute the lattice block.")
        app.listeners.append(self.on_result)
        # other extensions announce a change of their quantity (same values of sp, new numbers)
        self._qversion = {}
        self._has_quantity_hook = hasattr(app, "quantity_listeners")
        if self._has_quantity_hook:
            app.quantity_listeners.append(self._quantity_changed)
        app.book.nb.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed(), add="+")

    # .. quantities
    def quantity_names(self):
        return list(self.builtins) + [n for n in self.app.grid_quantities if n not in self.builtins]

    def _refresh_names(self):
        self.q_box.configure(values=self.quantity_names())

    def quantity(self, result, name):
        """(n, n, n) values, or None while they are being evaluated. Built-ins
        are evaluated here and cached per block; other extensions' quantities
        (app.grid_quantities) are evaluated on the worker thread in blocks of
        QUANTITY_BLOCK sites - their functions must not touch Tk - and cached per
        block and per new centre spectrum (their settings may have changed)."""
        fn = self.builtins.get(name) or self.app.grid_quantities.get(name)
        if fn is None:
            name = next(iter(self.builtins))
            fn = self.builtins[name]
        builtin = name in self.builtins
        version = self._qversion.get(name, 0) if self._has_quantity_hook else self.listener_calls
        key = (id(result), name, None if builtin else version)
        if key in self._qcache:
            return self._qcache[key]
        if len(self._qcache) > 64:
            self._qcache.clear()
        if builtin:
            self._qcache[key] = result.quantity(fn)
            if getattr(result, "quantity_error", ""):
                self.status.configure(text=f"{name}: {result.quantity_error}", foreground="#c60")
            return self._qcache[key]
        self._evaluate(result, name, fn, key)
        return None

    def _quantity_changed(self, name):
        """app.quantity_changed(name): drop the cached values of that quantity."""
        self._qversion[name] = self._qversion.get(name, 0) + 1
        for key in [k for k in self._qcache if k[1] == name]:
            del self._qcache[key]
        if self._q_pending is not None and self._q_pending[1] == name:
            self._q_pending = None
        if self.q_var.get() == name:
            self.redraw(["grid", "array"])

    def _evaluate(self, result, name, fn, key):
        if self._q_pending == key:
            return
        self._q_pending = key
        n = len(result.spectra)
        out = np.full(n, np.nan)            # filled by the worker, read here only when complete
        pos = [0]

        errors = [0, ""]                    # unexpected failures so far, the first one (G.site_values)

        def chunk(progress, cancelled):
            m0, m1 = pos[0], min(pos[0] + QUANTITY_BLOCK, n)
            n_bad, first = G.site_values(result.spectra, fn, out, range(m0, m1))
            errors[0] += n_bad
            errors[1] = errors[1] or first
            pos[0] = m1
            return m1

        def done(m1):
            if self._q_pending != key:
                return
            if m1 < n:
                if not self.running:
                    self.status.configure(text=f"evaluating {name}: {m1} / {n} sites")
                self.app.run_background(chunk, done=done)
                return
            self._q_pending = None
            self._qcache[key] = out.reshape(result.shape)
            if errors[0]:
                print(f"grid quantity {name!r} failed at {errors[0]} sites: {errors[1]}", file=sys.stderr)
                self.status.configure(text=f"{name}: failed at {errors[0]} of {n} sites ({errors[1]})",
                                      foreground="#c60")
            elif not self.running:
                self.status.configure(text=f"{n} sites", foreground="#666666")
            self.redraw()
        self.app.run_background(chunk, done=done)

    # .. computing
    def _grid_tab_shown(self):
        return self.app.book.current_key() in ("grid", "slices", "array")

    def is_stale(self):
        return self.result_key != grid_settings_key(self.se, self.app.settings)

    def _needs_reference(self):
        return self.ref_var.get() == REFERENCES[2]

    def _missing(self, settings):
        """Settings whose block is wanted and not in the store, in order."""
        wanted = [settings]
        if any_offset(settings) and self._needs_reference():
            wanted.append(centred_settings(self.se, settings))
        return [s for s in wanted if grid_settings_key(self.se, s) not in self.store]

    def start(self):
        """Show the block for the current settings, computing what is missing."""
        self.stop(quiet=True)
        settings = self.se.copy_settings(self.app.settings)
        missing = self._missing(settings)
        if not missing:
            self._show(settings)
            return
        self.run_id += 1
        rid = self.run_id
        target = missing[0]
        work = GridWork(self.se, target, self.cache)
        self.running = True
        what = "the block" if target is settings else "the block with every arm centred"
        self.status.configure(text=f"computing {what} ...", foreground="#666666")
        self.redraw()
        self._queue(rid, work, target, what)

    def _queue(self, rid, work, target, what):
        def fn(progress, cancelled):
            try:
                return work.step(cancelled)
            except Exception:
                return ("error", traceback.format_exc())
        self.handle = self.app.run_background(fn, done=lambda v: self._step_done(rid, work, target, what, v))

    def _step_done(self, rid, work, target, what, value):
        if rid != self.run_id:
            return
        kind = value[0]
        if kind == "running":
            _, done, total = value
            self.status.configure(text=f"{what}: step {done} / {total}")
            self._queue(rid, work, target, what)
            return
        self.running = False
        self.handle = None
        if kind == "error":
            self.status.configure(text="failed - see the console", foreground="#c60")
            print(value[1])
            return
        _, result, seconds = value
        self._store(grid_settings_key(self.se, target), result)
        self.timing.configure(text=(
            f"last block: {result.nodes_total} level calculations ({result.nodes_computed} new) and "
            f"{len(result.spectra)} site spectra in {seconds:.1f} s; {result.n_levels} levels per pair"
            + (f"; pairs computed site by site: {', '.join(result.direct_pairs)}" if result.direct_pairs else "")
            + (f"; {result.misses} levels outside the interpolation" if result.misses else "")))
        self.start()                     # the next missing block, or show

    def _store(self, key, result):
        self.store.pop(key, None)
        self.store[key] = result
        while len(self.store) > STORE_BLOCKS:
            self.store.pop(next(iter(self.store)))

    def _show(self, settings):
        key = grid_settings_key(self.se, settings)
        self.result = self.store[key]
        self.store[key] = self.store.pop(key)                      # most recently used last
        if any_offset(settings):
            self.ref = self.store.get(grid_settings_key(self.se, centred_settings(self.se, settings)))
        else:
            self.ref = self.result
        self.result_key, self.settings = key, settings
        self.status.configure(text=f"{len(self.result.spectra)} sites", foreground="#666666")
        self.redraw()

    def stop(self, quiet=False):
        self._q_pending = None                   # also ends an evaluation of another extension's quantity
        if self.handle is not None:
            self.handle.cancel()
        self.handle = None
        self.run_id += 1
        if self.running and not quiet:
            self.status.configure(text="stopped")
        self.running = False

    def _auto(self):
        return self.auto_var.get() and self._grid_tab_shown()

    def on_result(self, res):
        self.listener_calls += 1
        if self.is_stale() or (self.ref is None and self.result is not None and self._needs_reference()):
            if self._auto() or grid_settings_key(self.se, self.app.settings) in self.store:
                self.start()
            elif not self.running and self.result is not None:
                self.status.configure(text="settings changed - press Compute grid", foreground="#c60")
        self.redraw()

    def _tab_changed(self):
        if self._auto() and not self.running and (self.result is None or self.is_stale()):
            self.start()

    def _reference_changed(self):
        if self._needs_reference() and self.result is not None and self.ref is None \
                and not self.is_stale() and not self.running:
            self.start()
        self.redraw()

    # .. drawing
    def redraw(self, keys=("grid", "slices", "array")):
        for k in keys:
            self.app.book[k].mark_dirty()
        self.app.book.draw_visible()

    def _placeholder(self, fig, text=None):
        fig.clear()
        if text is None and self.result is not None and self._reference_missing():
            text = ("Computing the same block with every arm centred ..." if self.running else
                    "Press Compute grid to compute the same block with every arm centred.")
        elif text is None:
            text = ("Computing the lattice block ..." if self.running else
                    "Press Compute grid (settings column) to compute the lattice block.")
        fig.text(0.5, 0.5, text, ha="center", va="center", color="#666666")

    def _map_values(self):
        name = self.q_var.get()
        if name not in self.builtins and name not in self.app.grid_quantities:
            name = next(iter(self.builtins))
            self.q_var.set(name)
        values = self.quantity(self.result, name)
        if values is None:
            return name, None, None
        if self.ref is None:
            ref_values = np.full_like(values, np.nan)
        elif self.ref is self.result:
            ref_values = values
        else:
            ref_values = self.quantity(self.ref, name)
            if ref_values is None:
                return name, None, None
        return name, values, ref_values

    def _plane_index(self):
        try:
            s = int(self.s_var.get())
        except Exception:
            s = 0
        s = min(max(s, -G.GRID_HALF), G.GRID_HALF)
        self.s_var.set(s)
        return s

    def _save_map(self):
        return f"lattice_grid_{G.PAIRS[PLANES[self.plane_var.get()]]}"

    def _reference_missing(self):
        return self.ref is None and self._needs_reference()

    def _draw_map(self, fig):
        if self.result is None or self._reference_missing():
            return self._placeholder(fig)
        name, values, ref_values = self._map_values()
        if values is None:
            return self._placeholder(fig, f"Evaluating {name} at every site ...")
        draw_maps(fig, self.se, self.result, self.ref, values, ref_values, name, self.ref_var.get(),
                  PLANES[self.plane_var.get()], self._plane_index(), self.all_var.get(), self.settings)

    def _draw_slices(self, fig):
        if self.result is None:
            return self._placeholder(fig)
        draw_slices(fig, self.se, self.result, self.app.settings["glob"]["width_hz"], self.follow_var.get(),
                    self.settings)

    def _draw_array(self, fig):
        if self.result is None or self._reference_missing():
            return self._placeholder(fig)
        name, values, ref_values = self._map_values()
        if values is None:
            return self._placeholder(fig, f"Evaluating {name} at every site ...")
        draw_array(fig, self.se, self.result, self.ref, self.app.settings["glob"]["width_hz"], values,
                   ref_values, name, self.ref_var.get(), self.settings, log=self.log_var.get())


def attach(app):
    return GridView(app)
