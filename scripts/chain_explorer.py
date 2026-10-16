"""Atom-chain window: 2-7 neutral atoms on neighbouring x sites through the
spin-echoed two-leg collisional gate, with the overlap of neighbouring packets.

Sequence (Du 2025, Sec. 4.2.2.3): pi/2 - leg 1 - pi - leg 2 - pi/2, a leg
being map to the collisional states |3,+m> / |3,-m>, rotate the EOM arm out
to the hold voltage, hold, rotate back, map back. The |3,+m> packet of each
atom moves +x and the |3,-m> packet -x, so the |3,+m> packet of atom j meets
the |3,-m> packet of atom j+1 between the two sites and the pair picks up a
collisional phase. The physics is ``lattice_sim.gates.chain`` (the Phase 4
pair model of ``gates.collision`` repeated along the chain; tested in
``tests/test_chain.py``).

Window. Left: the settings, each with a one-line explanation. Right: tabs

* Sequence - the chain along x (one row per atom, packets coloured by spin
  component), the densities at one meeting point, and the EOM voltage, the
  relative overlap of a neighbour pair and its accumulated collisional phase
  against time, with the pulses marked. A time slider and Play/Pause step
  through the precomputed sequence (only the moving artists are redrawn);
* Ramsey signal - the fringe of the final pi/2 phase for atoms with 0, 1 and
  2 interacting neighbours (cos^n(Phi/2) rule and an explicit state-vector
  simulation of the chain), and the total collisional phase per pair and the
  fringe contrast against the hold voltage at the set hold time.

Below the tabs: the numbers at the hold, the phases, the 7-state check of
``CollisionModel.evolve`` (first-principles models) and a table per atom.

Usage: python scripts/chain_explorer.py            # standalone window
       (or the "Atom chain" group of scripts/sideband_explorer.py)
"""

import math
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lattice_sim.constants import A0  # noqa: E402
from lattice_sim.gates import chain as ch  # noqa: E402

# ------------------------------------------------------------------ settings
PLUS_COLOUR = "#1f77b4"          # |3,+m>
MINUS_COLOUR = "#d62728"         # |3,-m>
STORAGE_COLOUR = "#7f7f7f"       # qubit states at the site
N_COLOURS = {0: "#7f7f7f", 1: "#2e7d32", 2: "#9467bd"}
PULSE_COLOUR = "#222222"
MAP_COLOUR = "#9a9a9a"
CURSOR_COLOUR = "#e07b00"
OVERLAP_FLOOR = 1e-6
DEFAULT_HOLD_V = {3: 4150.0, 1: 3300.0}
PLAYBACK_S = {"5 s": 5.0, "10 s": 10.0, "20 s": 20.0, "40 s": 40.0}
TICK_MS = 30
DEBOUNCE_MS = 250
ZOOM_MIN_HALF_NM = 300.0
CONTRAST_PHI_MAX = 6 * np.pi     # contrast vs hold voltage is drawn up to this total phase

MODEL_PHASE4 = "Phase 4: first principles, measured ER, 10 kHz"
MODEL_CAL = "Calibration files: ER316, 10 kHz (|3,±3⟩ only)"

ASSUMPTIONS = (
    "Assumptions: each atom starts in its 3D vibrational ground state; its two spin components are "
    "harmonic-oscillator Gaussians that follow the trap minima adiabatically (no motional excitation, "
    "1D along x); the transverse overlap is that of identical Gaussians at the set y, z frequencies; "
    "U/h = 2ħ a O / m (Phase 4, collision.py), spin exchange neglected in the phase; ideal "
    "instantaneous π/2, π and map pulses; no field, intensity or pointing noise; identical legs, so the "
    "light-shift phases cancel; a homogeneous chain (every neighbour pair identical); only nearest "
    "neighbours overlap. Assumed map: |4,+1⟩ → |3,+m⟩ (moves +x), |3,−1⟩ → |3,−m⟩.")


def state_label(m, sign):
    return f"|3,{'+' if sign > 0 else '−'}{m}⟩"


# ------------------------------------------------------------------ drawing
def _vlines(ax, res, label_pulses):
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for name, t in res["pulses"]:
        ax.axvline(t / 1e3, color=PULSE_COLOUR, lw=0.8, ls="--", zorder=1)
        if label_pulses:
            ax.text(t / 1e3, 1.01, name, transform=trans, ha="center", va="bottom", fontsize=7.5)
    for t in res["maps"]:
        ax.plot([t / 1e3], [0.0], marker="^", ms=4, color=MAP_COLOUR, transform=trans, clip_on=False, zorder=3)


def draw_sequence(fig, res, labplot):
    """Build the Sequence figure; returns the animated artists for update_frame."""
    s = res["settings"]
    n = s.n_atoms
    fig.clear()
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0], width_ratios=[1.0, 1.0, 1.0])
    ax_chain = fig.add_subplot(gs[0, :2])
    ax_zoom = fig.add_subplot(gs[0, 2])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_o = fig.add_subplot(gs[1, 1])
    ax_p = fig.add_subplot(gs[1, 2])
    f = res["frames"]
    a_um = res["spacing_m"] * 1e6
    sites = ch.site_positions(n, res["spacing_m"]) * 1e6
    peak = 1.0 / (np.sqrt(np.pi) * min(f["s_plus"].min(), f["s_minus"].min()) * 1e6)   # 1/um, storage
    row = 1.15 * peak
    base = (n - 1 - np.arange(n)) * row
    for j in range(n):
        ax_chain.axhline(base[j], color="#cccccc", lw=0.6, zorder=0)
    for x in sites:
        ax_chain.axvline(x, color="#dddddd", lw=0.6, ls=":", zorder=0)
    ax_chain.set_xlim(sites[0] - 0.8 * a_um, sites[-1] + 0.8 * a_um)
    ax_chain.set_ylim(-0.08 * row, base[0] + row)
    ax_chain.set_yticks(base)
    ax_chain.set_yticklabels([f"atom {j + 1}" for j in range(n)])
    m = s.m
    art = dict(plus=[], minus=[])
    for j in range(n):
        lp, = ax_chain.plot([], [], color=PLUS_COLOUR, lw=1.2, animated=True)
        lm, = ax_chain.plot([], [], color=MINUS_COLOUR, lw=1.2, animated=True)
        art["plus"].append(lp)
        art["minus"].append(lm)
    # legend proxies (static)
    ax_chain.plot([], [], color=PLUS_COLOUR, lw=1.2, label=f"{state_label(m, +1)} (moves +x)")
    ax_chain.plot([], [], color=MINUS_COLOUR, lw=1.2, label=f"{state_label(m, -1)} (moves −x)")
    ax_chain.plot([], [], color=STORAGE_COLOUR, lw=1.2, label="qubit states |3,−1⟩, |4,+1⟩ at the site")
    labplot.legend(ax_chain, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncols=3, frameon=False)
    ax_chain.set_ylim(-0.08 * row, base[0] + 1.12 * row)
    # filled in only for a saved figure (the window shows these values in the bar above the figure)
    art["time"] = ax_chain.text(0.005, 0.99, "", transform=ax_chain.transAxes, ha="left", va="top", fontsize=7)
    labplot.note(ax_chain, f"row height {row:.1f} /µm", "top-right")
    labplot.finish(ax_chain, "Position along x (µm), chain centred on x = 0", "Density (1/µm), one row per atom",
                   f"Densities of the two spin components of {n} atoms on neighbouring sites "
                   f"(spacing {a_um:.3f} µm)")
    ax_chain.grid(False)
    ax_chain.grid(True, axis="x", alpha=0.3)

    # one meeting point
    d_hold_nm = res["hold"]["separation_m"] * 1e9
    w_nm = max(res["hold"]["widths_m"]) * 1e9
    half = max(ZOOM_MIN_HALF_NM, d_hold_nm / 2 + 4 * w_nm)
    art["zoom_half_nm"] = half
    zp, = ax_zoom.plot([], [], color=PLUS_COLOUR, lw=1.2, animated=True, label=f"atom j, {state_label(m, +1)}")
    zm, = ax_zoom.plot([], [], color=MINUS_COLOUR, lw=1.2, animated=True, label=f"atom j+1, {state_label(m, -1)}")
    art["zoom"] = (zp, zm)
    art["zoom_text"] = ax_zoom.text(0.02, 0.985, "", transform=ax_zoom.transAxes, ha="left", va="top", fontsize=7)
    ax_zoom.set_xlim(-half, half)
    ax_zoom.set_ylim(0, 0.62 * peak)
    labplot.legend(ax_zoom, loc="upper right")
    labplot.finish(ax_zoom, "Position from the midpoint of sites j and j+1 (nm)", "Density (1/µm)",
                   "Densities between two neighbouring sites")

    t_ms, theta_d, rel_d, phase_d = decimated(res)
    ax_v.plot(t_ms, ch.theta_to_voltage(theta_d), color="#222222", lw=1.0)
    ax_v.set_xlim(0, t_ms[-1])
    ax_v.set_ylim(-150, 5600)
    _vlines(ax_v, res, True)
    labplot.finish(ax_v, "Time in the sequence (ms)", "EOM voltage (V)", None)
    ax_v.set_title("EOM voltage and pulses", fontsize=8, pad=11)
    labplot.note(ax_v, f"hold {s.hold_v:.0f} V, {res['hold_us']:.0f} µs\n▲ map pulses (ideal)", "top-left")

    ax_o.semilogy(t_ms, np.maximum(rel_d, OVERLAP_FLOOR), color="#222222", lw=1.0)
    ax_o.set_xlim(0, t_ms[-1])
    ax_o.set_ylim(OVERLAP_FLOOR, 100.0)
    _vlines(ax_o, res, False)
    labplot.finish(ax_o, "Time in the sequence (ms)", "Relative overlap",
                   "Overlap of a neighbour pair, relative to V_π")
    ax_o.yaxis.set_minor_locator(NullLocator())
    ax_o.grid(False, which="minor")
    labplot.note(ax_o, f"at the hold {res['hold']['rel_overlap']:.3g}\nclipped at {OVERLAP_FLOOR:.0e}", "top-left")

    ax_p.plot(t_ms, phase_d / np.pi, color="#222222", lw=1.0)
    for y in (0.5, 1.0):
        ax_p.axhline(y, color="#999999", lw=0.6, ls=":")
    ax_p.set_xlim(0, t_ms[-1])
    top = max(1.25, 1.1 * res["phase"][-1] / np.pi)
    ax_p.set_ylim(0, top)
    _vlines(ax_p, res, False)
    labplot.finish(ax_p, "Time in the sequence (ms)", "Collisional phase (π)",
                   "Accumulated phase of a neighbour pair")
    labplot.note(ax_p, f"leg 1 {res['phi1'] / np.pi:.3f} π\nleg 2 {res['phi2'] / np.pi:.3f} π\n"
                       "every pair identical", "bottom-right")

    cursors = []
    for ax, y in ((ax_v, ch.theta_to_voltage(f["theta"])), (ax_o, np.maximum(f["rel_overlap"], OVERLAP_FLOOR)),
                  (ax_p, f["phase"] / np.pi)):
        ln = ax.axvline(0, color=CURSOR_COLOUR, lw=1.0, animated=True)
        dot, = ax.plot([], [], "o", color=CURSOR_COLOUR, ms=4, animated=True)
        cursors.append((ln, dot, y))
    art["cursors"] = cursors
    art["sites"] = sites
    art["base"] = base
    art["animated"] = (art["plus"] + art["minus"] + [zp, zm]
                       + [c[0] for c in cursors] + [c[1] for c in cursors])
    return art


def contrast_and_shift(f, tol=1e-9):
    """Signed contrast and fringe shift (deg) of a coherence factor F: the
    fringe is (1 - |F| cos(phase + arg F)) / 2; a real F keeps its sign."""
    f = complex(f)
    if abs(f) < 5e-5:
        return 0.0, 0.0
    if abs(f.imag) <= tol * max(1.0, abs(f)):
        return f.real, 0.0
    return abs(f), -math.degrees(math.atan2(f.imag, f.real))


def segment_at(res, t_us):
    for name, t0, t1 in res["segments"]:
        if t0 <= t_us <= t1:
            return name
    return "qubit states at the sites"


def decimated(res, n=4001):
    """The time series on a coarser grid for drawing, keeping every leg breakpoint."""
    t = res["t_us"]
    g = res["settings"].gap_us
    starts = [g / 2, g / 2 + res["leg_us"] + g]
    keep = np.unique(np.concatenate([np.linspace(t[0], t[-1], n)] + [t0 + res["leg_t"] for t0 in starts]))
    keep = keep[(keep >= t[0]) & (keep <= t[-1])]
    return (keep / 1e3, np.interp(keep, t, res["theta"]), np.interp(keep, t, res["rel_overlap"]),
            np.interp(keep, t, res["phase"]))


def update_frame(art, res, i, texts=False):
    """Set the moving artists to frame i (no drawing). Returns the readout
    strings (time and segment; meeting point); with texts=True they are also
    written into the figure (for a saved image)."""
    f = res["frames"]
    s = res["settings"]
    n = s.n_atoms
    xp, xm, sp, sm = f["x_plus"][i] * 1e6, f["x_minus"][i] * 1e6, f["s_plus"][i] * 1e6, f["s_minus"][i] * 1e6
    u = np.linspace(-4.0, 4.0, 121)
    in_leg = bool(f["in_leg"][i])
    for j in range(n):
        c = art["sites"][j]
        b = art["base"][j]
        if in_leg:
            xs = c + xp + sp * u
            art["plus"][j].set_data(xs, b + 0.5 * ch.gaussian_density(xs, c + xp, sp))
            art["plus"][j].set_color(PLUS_COLOUR)
            xs = c - xm + sm * u
            art["minus"][j].set_data(xs, b + 0.5 * ch.gaussian_density(xs, c - xm, sm))
            art["minus"][j].set_visible(True)
        else:
            xs = c + xp + sp * u
            art["plus"][j].set_data(xs, b + ch.gaussian_density(xs, c + xp, sp))
            art["plus"][j].set_color(STORAGE_COLOUR)
            art["minus"][j].set_visible(False)
    t = f["t_us"][i]
    v = float(ch.theta_to_voltage(f["theta"][i]))
    line_t = f"t = {t / 1e3:.3f} ms   {segment_at(res, t)}   θ = {np.degrees(f['theta'][i]):.1f}° ({v:.0f} V)"
    half = art["zoom_half_nm"]
    a_nm = res["spacing_m"] * 1e9
    zp, zm = art["zoom"]
    if in_leg:
        cp, cm_ = xp * 1e3 - a_nm / 2, a_nm / 2 - xm * 1e3
        xs = np.linspace(-half, half, 601)
        zp.set_data(xs, 0.5 * ch.gaussian_density(xs * 1e-3, cp * 1e-3, sp))
        zm.set_data(xs, 0.5 * ch.gaussian_density(xs * 1e-3, cm_ * 1e-3, sm))
        d = f["separation"][i]
        dtxt = f"{d * 1e9:.0f} nm" if d < 2e-6 else f"{d * 1e6:.2f} µm"
        line_z = (f"separation {dtxt}\nwidths {sp * 1e3:.0f} / {sm * 1e3:.0f} nm\n"
                  f"relative overlap {f['rel_overlap'][i]:.2e}\nU/h {f['u_hz'][i]:.3g} Hz")
    else:
        zp.set_data([], [])
        zm.set_data([], [])
        line_z = "qubit states at the sites (θ = 0)"
    for ln, dot, y in art["cursors"]:
        ln.set_xdata([t / 1e3, t / 1e3])
        dot.set_data([t / 1e3], [y[i]])
    art["time"].set_text(line_t if texts else "")
    art["zoom_text"].set_text(line_z if texts else "")
    return line_t, line_z


def draw_ramsey(fig, res, scan, labplot):
    """The Ramsey tab: (a) fringes, (b) contrast vs total phase per pair,
    (c) total phase vs hold voltage, (d) contrast vs hold voltage."""
    from scipy.interpolate import CubicSpline
    fig.clear()
    (a, b), (c, d) = fig.subplots(2, 2)
    s = res["settings"]
    ph_deg = np.degrees(res["scan_phases"])
    total = res["phi1"] + res["phi2"]
    names = {0: "no neighbour (n = 0)", 1: "end atom (n = 1)", 2: "interior atom (n = 2)"}
    shown = (0, 1, 2) if s.n_atoms >= 3 else (0, 1)
    for nb in shown:
        a.plot(ph_deg, res["fringes_ref"][nb], color=N_COLOURS[nb], lw=1.1, ls="--" if nb == 0 else "-",
               label=f"cos^n(Φ/2) rule, {names[nb]}")
    nbs = res["neighbours"]
    for nb, marker in ((1, "o"), (2, "s")):
        idx = np.flatnonzero(nbs == nb)
        if idx.size:
            k = int(idx[0])
            a.plot(ph_deg, res["fringes_sim"][k], marker, ms=3.5, mfc="none", color=N_COLOURS[nb],
                   label=f"state-vector simulation, atom {k + 1}")
    a.set_xlim(0, 360)
    a.set_xticks(np.arange(0, 361, 90))
    a.set_ylim(-0.02, 1.2)
    labplot.legend(a, loc="lower center")
    labplot.finish(a, "Phase of the final π/2 pulse (deg)", "Probability in |4,+1⟩",
                   "Ramsey fringe after the gate sequence")
    labplot.note(a, f"Φ = {total / np.pi:.3f} π per pair, {s.n_atoms} atoms\n"
                    "all end atoms alike, all interior atoms alike", "top-left")

    # contrast vs the total phase per pair: the rule and the state-vector simulation
    phis = np.linspace(0, 4 * np.pi, 401)
    pts = np.linspace(0, 4 * np.pi, 17)
    sim = np.array([ch.chain_fringes(3, p / 2, p / 2, np.array([0.0, np.pi])) for p in pts])
    sim_c = sim[:, :, 1] - sim[:, :, 0]                  # P(pi) - P(0) = contrast, per atom
    for nb in shown:
        b.plot(phis / np.pi, ch.neighbour_contrast(nb, phis), color=N_COLOURS[nb], lw=1.1,
               ls="--" if nb == 0 else "-", label=f"cos^n(Φ/2), {names[nb]}")
    b.plot(pts / np.pi, sim_c[:, 0], "o", ms=3.5, mfc="none", color=N_COLOURS[1],
           label="state-vector simulation, 3 atoms: atom 1")
    b.plot(pts / np.pi, sim_c[:, 1], "s", ms=3.5, mfc="none", color=N_COLOURS[2],
           label="state-vector simulation, 3 atoms: atom 2")
    if total <= 4 * np.pi:
        b.axvline(total / np.pi, color=CURSOR_COLOUR, lw=0.9, label="current Φ")
    b.axhline(0.0, color="#999999", lw=0.6)
    b.set_xlim(0, 4)
    b.set_ylim(-1.1, 1.75)
    labplot.legend(b, loc="upper center", ncols=2)
    labplot.finish(b, "Total collisional phase per pair Φ (π)", "Fringe contrast",
                   "Fringe contrast vs total collisional phase per pair")

    if scan is None:
        for ax in (c, d):
            ax.text(0.5, 0.5, "computing the hold-voltage scan ...", transform=ax.transAxes, ha="center",
                    va="center", color="#666666")
            labplot.finish(ax, "Hold voltage (V)", "")
        return
    v = scan["volts"]
    c.plot(v, scan["total"] / np.pi, color="#222222", lw=1.1)
    c.plot([s.hold_v], [total / np.pi], "o", color=CURSOR_COLOUR, ms=5, label="current setting")
    c.axhline(1.0, color="#999999", lw=0.6, ls=":")
    c.set_yscale("log")
    c.yaxis.set_minor_locator(NullLocator())
    c.set_ylim(1e-3, max(10.0, 1.5 * float(np.nanmax(scan["total"])) / np.pi))
    c.set_xlim(v[0], v[-1])
    labplot.legend(c, loc="upper left")
    labplot.finish(c, "Hold voltage (V)", "Φ (π)", "Total collisional phase per pair vs hold voltage")
    labplot.note(c, f"hold {scan['hold_us']:.0f} µs in each leg, fixed\ndotted: Φ = π", "bottom-right")

    # contrast vs hold voltage up to Φ = CONTRAST_PHI_MAX, from a spline of log Φ(V)
    log_phi = CubicSpline(v, np.log(np.maximum(scan["total"], 1e-30)))
    above = np.flatnonzero(scan["total"] > CONTRAST_PHI_MAX)
    v_hi = v[above[0]] if above.size else v[-1]
    vf = np.linspace(v[0], v_hi, 1500)
    phi_f = np.exp(log_phi(vf))
    for nb in shown[1:]:
        d.plot(vf, ch.neighbour_contrast(nb, phi_f), color=N_COLOURS[nb], lw=1.1, label=names[nb])
    if v[0] <= s.hold_v <= v_hi:
        d.axvline(s.hold_v, color=CURSOR_COLOUR, lw=0.9, label="current hold voltage")
    d.axhline(0.0, color="#999999", lw=0.6)
    d.set_xlim(v[0], v_hi)
    d.set_ylim(-1.1, 1.45)
    labplot.legend(d, loc="upper right", ncols=3)
    labplot.finish(d, "Hold voltage (V)", "Fringe contrast cos^n(Φ/2)", "Fringe contrast vs hold voltage")
    labplot.note(d, f"hold {scan['hold_us']:.0f} µs fixed; shown up to Φ = {CONTRAST_PHI_MAX / np.pi:.0f} π",
                 "bottom-left")


# ------------------------------------------------------------------ worker
class Worker:
    """run_background(fn, done, progress) on a thread of our own (standalone use);
    the same contract as the sideband explorer's App.run_background."""

    def __init__(self, widget):
        self.widget = widget
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()
        widget.after(50, self._poll)

    def run_background(self, fn, done=None, progress=None):
        handle = type("Job", (), {})()
        handle.flag = False
        handle.cancel = lambda: setattr(handle, "flag", True)
        self.jobs.put((fn, done, progress, handle))
        return handle

    def close(self):
        self.alive = False
        self.jobs.put(None)

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            fn, done, progress, handle = job

            def report(i, n, progress=progress):
                if progress is not None:
                    self.results.put((progress, (i, n)))
            try:
                value = fn(report, lambda handle=handle: handle.flag)
            except Exception:
                traceback.print_exc()
                continue
            if done is not None:
                self.results.put((done, (value,)))

    def _poll(self):
        if not self.alive:
            return
        try:
            while True:
                fn, args = self.results.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        try:
            self.widget.after(50, self._poll)
        except Exception:
            pass


# ------------------------------------------------------------------ window
class ChainWindow:
    """The chain window: a Toplevel of ``master`` (or ``master`` itself when
    standalone). defaults: the sideband explorer's x-pair settings (see
    ``explorer_chain.defaults_from_settings``); run_background: the explorer's
    worker (None: a worker of our own)."""

    def __init__(self, master=None, defaults=None, run_background=None):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk
        import labplot
        self.tk, self.ttk, self.labplot = tk, ttk, labplot
        if master is None:
            self.top = tk.Tk()
            labplot.apply_style()
            self.top.geometry("1520x960")
        else:
            self.top = tk.Toplevel(master)
            self.top.geometry("1520x960")
        self.top.title("Atom chain: spin-echoed collisional gate - lattice-sim")
        self.worker = None
        if run_background is None:
            self.worker = Worker(self.top)
            run_background = self.worker.run_background
        self.run_background = run_background
        self.defaults = defaults
        self.models = {}               # (model, m, h) -> CollisionModel
        self.building = set()
        self.result = None
        self.scan = None
        self.check = None
        self.art = None
        self.frame = 0
        self.playing = False
        self._gen = 0
        self._after = None
        self._tick_after = None
        self._acc = 0.0
        self._last_tick = 0.0
        self._bg = None
        self._saving = False
        self._loading = False
        self.closed = False

        base = tkfont.nametofont("TkDefaultFont")
        self.small = base.copy()
        self.small.configure(size=max(7, base.cget("size") - 1))
        self.bold = base.copy()
        self.bold.configure(weight="bold")
        self.wrap = 330

        outer = ttk.Panedwindow(self.top, orient="horizontal")
        outer.pack(fill="both", expand=True)
        left = self._scrollable(outer, 360)
        right = ttk.Frame(outer)
        outer.add(left[0], weight=0)
        outer.add(right, weight=1)
        self._build_settings(left[1])

        # plots
        read = ttk.Frame(right, padding=(6, 4))
        read.pack(side="bottom", fill="x")
        nb = ttk.Notebook(right)
        nb.pack(side="top", fill="both", expand=True)
        self.book = labplot.PlotBook(nb)
        figs = str(REPO / "figures")
        self.p_seq = self.book.add("sequence", "Sequence", draw=self._draw_sequence, figsize=(10, 6.2),
                                   save_name=lambda: f"chain_sequence_{self.settings().n_atoms}atoms",
                                   save_dir=figs)
        self.p_ram = self.book.add("ramsey", "Ramsey signal", draw=self._draw_ramsey, figsize=(10, 6.2),
                                   save_name=lambda: f"chain_ramsey_{self.settings().n_atoms}atoms",
                                   save_dir=figs)
        self._wrap_save(self.p_seq)
        self.p_seq.canvas.mpl_connect("draw_event", self._on_draw)
        c = self.p_seq.controls
        self.play_btn = ttk.Button(c, text="Play", width=7, command=self._toggle_play)
        self.play_btn.pack(side="left", padx=(4, 4))
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(c, from_=0, to=1, variable=self.time_var, command=self._slid)
        self.time_scale.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Label(c, text="sequence plays in").pack(side="left", padx=(6, 2))
        self.speed_var = tk.StringVar(value="10 s")
        ttk.Combobox(c, textvariable=self.speed_var, values=list(PLAYBACK_S), width=6,
                     state="readonly").pack(side="left")
        self.time_label = ttk.Label(self.p_seq.controls, text="", anchor="w", foreground="#444444")
        self.time_label.pack(side="bottom", fill="x", padx=6, before=self.play_btn)

        self.summary = ttk.Label(read, text="", justify="left", wraplength=1080)
        self.summary.pack(fill="x")
        cols = ("atom", "neighbours", "rule", "sim", "shift")
        self.tree = ttk.Treeview(read, columns=cols, show="headings", height=7)
        for col, text, w in (("atom", "atom", 70), ("neighbours", "interacting neighbours n", 170),
                             ("rule", "fringe contrast, cos^n(Φ/2) rule", 230),
                             ("sim", "fringe contrast, state-vector simulation", 260),
                             ("shift", "fringe phase shift (deg)", 170)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="e" if col != "atom" else "w")
        self.tree.pack(fill="x", pady=(4, 2))
        self.status = ttk.Label(self.top, text="", foreground="#666666", anchor="w", padding=(6, 2))
        self.status.pack(fill="x", side="bottom")

        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._submit()

    # .. layout ---------------------------------------------------------
    def _scrollable(self, parent, width):
        tk, ttk = self.tk, self.ttk
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
            under = self.top.winfo_containing(event.x_root, event.y_root)
            if under is not None and str(under).startswith(str(outer)):
                canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")
        self.top.bind("<MouseWheel>", wheel, add="+")
        return outer, inner

    def _help(self, parent, text):
        lbl = self.ttk.Label(parent, text=text, font=self.small, foreground="#666666", wraplength=self.wrap - 20,
                             justify="left")
        lbl.pack(fill="x")
        return lbl

    def _row(self, parent, label, unit, lo, hi, value, digits, help_text):
        """Label, typed value, unit, slider and a one-line explanation."""
        tk, ttk = self.tk, self.ttk
        var = tk.DoubleVar(value=value)
        text = tk.StringVar(value=f"{value:.{digits}f}")
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(5, 0))
        top = ttk.Frame(frame)
        top.pack(fill="x")
        name = ttk.Label(top, text=label)
        name.pack(side="left")
        ttk.Label(top, text=unit, width=5).pack(side="right")
        entry = ttk.Entry(top, textvariable=text, width=9, justify="right")
        entry.pack(side="right")
        row = type("Row", (), {})()
        row.var, row.text, row.lo, row.hi, row.digits = var, text, lo, hi, digits

        def slid(_=None):
            text.set(f"{var.get():.{digits}f}")
            self._changed()

        def typed(_=None):
            try:
                v = float(text.get())
            except ValueError:
                text.set(f"{var.get():.{digits}f}")
                return
            v = min(max(v, lo), hi)
            if abs(v - var.get()) > 10 ** -(digits + 1):
                var.set(v)
                text.set(f"{v:.{digits}f}")
                self._changed()
            else:
                text.set(f"{v:.{digits}f}")
        entry.bind("<Return>", typed)
        entry.bind("<FocusOut>", typed)
        row.scale = ttk.Scale(frame, from_=lo, to=hi, variable=var, command=slid)
        row.scale.pack(fill="x")
        row.entry = entry
        row.help = self._help(frame, help_text) if help_text else None

        def get():
            return float(var.get())

        def set_(v):
            var.set(v)
            text.set(f"{v:.{digits}f}")

        def enable(on):
            for w in (row.scale, row.entry):
                w.state(["!disabled"] if on else ["disabled"])
        row.get, row.set, row.enable = get, set_, enable
        return row

    def _build_settings(self, body):
        tk, ttk = self.tk, self.ttk
        wrap = self.wrap
        ttk.Label(body, text=("A line of neutral atoms on neighbouring x sites through the spin-echoed "
                              "two-leg collisional gate: π/2, leg 1, π, leg 2, π/2. In each leg the "
                              "|3,+m⟩ packet of an atom moves +x and its |3,−m⟩ packet −x; at the hold the "
                              "|3,+m⟩ packet of atom j overlaps the |3,−m⟩ packet of atom j+1. Type a value "
                              "and press Enter, or drag."),
                  wraplength=wrap, justify="left", font=self.small).pack(fill="x", pady=(2, 6))

        grp = ttk.LabelFrame(body, text="Chain and states", padding=6)
        grp.pack(fill="x")
        r = ttk.Frame(grp)
        r.pack(fill="x")
        ttk.Label(r, text="Atoms:").pack(side="left")
        self.n_var = tk.IntVar(value=5)
        for n in range(2, 8):
            ttk.Radiobutton(r, text=str(n), value=n, variable=self.n_var, command=self._changed).pack(side="left",
                                                                                                    padx=2)
        self._help(grp, "End atoms have one interacting neighbour, interior atoms two.")
        ttk.Label(grp, text="Collisional state pair:").pack(anchor="w", pady=(6, 0))
        self.m_var = tk.IntVar(value=3)
        for m in (3, 1):
            ttk.Radiobutton(grp, text=f"{state_label(m, +1)} and {state_label(m, -1)}", value=m, variable=self.m_var,
                            command=self._pair_chosen).pack(anchor="w", padx=12)
        self._help(grp, "Both in F = 3 with opposite m_F (F = 4 is excluded: hyperfine-changing collisions "
                        "eject the atoms). The choice is open. Qubit states |3,−1⟩, |4,+1⟩ at 1.393 G; the "
                        "hold voltage is reset to the pair's default on a change.")

        grp = ttk.LabelFrame(body, text="Lattice model", padding=6)
        grp.pack(fill="x", pady=(8, 0))
        self.model_var = tk.StringVar(value=MODEL_PHASE4)
        self.model_box = ttk.Combobox(grp, textvariable=self.model_var, state="readonly", width=52)
        self.model_box.pack(fill="x")
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self._changed())
        self._refresh_models()
        hrow = ttk.Frame(grp)
        hrow.pack(fill="x", pady=(6, 0))
        ttk.Label(hrow, text="Ellipticity handedness:").pack(side="left")
        h0 = int(self.defaults["handedness"]) if self.defaults else -1
        self.h_var = tk.IntVar(value=h0)
        for h in (-1, 1):
            ttk.Radiobutton(hrow, text=f"{h:+d}", value=h, variable=self.h_var,
                            command=self._changed).pack(side="left", padx=4)
        self._help(grp, "Packet positions and trap frequencies vs θ. First principles: "
                        "collision.build_collision_model (~30 s per pair and handedness the first time, then "
                        "cached in data/). Calibration files: the |3,−3⟩ track and the optimizer's |4,−3⟩ "
                        "stand-in, the source of the lab's overlap-vs-voltage table (|3,±3⟩ only; no "
                        "handedness).")

        grp = ttk.LabelFrame(body, text="Gate timing", padding=6)
        grp.pack(fill="x", pady=(8, 0))
        self.r_hold_v = self._row(grp, "Hold voltage", "V", 2900.0, 5400.0, DEFAULT_HOLD_V[3], 0,
                                  "EOM voltage of the hold (θ = π V / 5400 V). Sets how far the packets of "
                                  "neighbouring atoms overlap: the gate's tuning knob.")
        ttk.Label(grp, text="Ramp:").pack(anchor="w", pady=(6, 0))
        self.ramp_var = tk.StringVar(value="robust")
        ttk.Radiobutton(grp, text="Phase 4 robust ramp, stopped at the hold", value="robust",
                        variable=self.ramp_var, command=self._changed).pack(anchor="w", padx=12)
        ttk.Radiobutton(grp, text="smooth sin² ramp of the set duration", value="smooth",
                        variable=self.ramp_var, command=self._changed).pack(anchor="w", padx=12)
        self.r_ramp = self._row(grp, "Smooth ramp duration", "µs", 50.0, 5000.0, 1000.0, 0,
                                "Each way, for the smooth ramp. The robust ramps (R3 5 ms for |3,±3⟩, R1 8 ms "
                                "for |3,±1⟩) keep their own timing up to the hold voltage.")
        self.lock_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="set the hold time for π/2 per leg", variable=self.lock_var,
                        command=self._changed).pack(anchor="w", pady=(6, 0))
        self.r_hold = self._row(grp, "Hold time", "µs", 0.0, 5000.0, 500.0, 0,
                                "Time at the hold voltage in each leg. When the box above is ticked it is "
                                "solved so that ramps plus hold give π/2; it is 0 when the ramps alone exceed "
                                "π/2.")
        self.r_gap = self._row(grp, "Time between the legs", "µs", 0.0, 2000.0, 200.0, 0,
                               "The π pulse sits in the middle; half of it is also kept before leg 1 and "
                               "after leg 2 (display only, no free-evolution phase is modelled).")

        grp = ttk.LabelFrame(body, text="Interaction", padding=6)
        grp.pack(fill="x", pady=(8, 0))
        self.sc_var = tk.StringVar(value="uniform")
        ttk.Radiobutton(grp, text="uniform scattering length (set below)", value="uniform", variable=self.sc_var,
                        command=self._changed).pack(anchor="w")
        ttk.Radiobutton(grp, text="DIS estimate of the channel lengths (Chin et al. 2004)", value="DIS",
                        variable=self.sc_var, command=self._changed).pack(anchor="w")
        self.r_a = self._row(grp, "Scattering length", "a0", 500.0, 4000.0, ch.DEFAULT_SCATTERING_A0, 0,
                             "2800 a0 is Du's assumption for F = 3 + F = 3; the value near 1.39 G is not known. "
                             "DIS gives 1596 a0 for |3,±3⟩ and 1866 a0 for |3,±1⟩ (diagonal).")
        self.r_fy = self._row(grp, "Transverse trap frequency y", "kHz", 3.0, 15.0, ch.TRANSVERSE_HZ[0] / 1e3, 2,
                              "y and z enter only through the transverse overlap. Phase 4: 8.0 / 6.25 kHz "
                              "(assumed); the lab's overlap tool used 7 / 7 kHz.")
        self.r_fz = self._row(grp, "Transverse trap frequency z", "kHz", 3.0, 15.0, ch.TRANSVERSE_HZ[1] / 1e3, 2,
                              "")
        brow = ttk.Frame(body)
        brow.pack(fill="x", pady=8)
        ttk.Button(brow, text="Reset", command=self._reset).pack(side="left")
        ttk.Label(body, text=ASSUMPTIONS, wraplength=wrap, justify="left", font=self.small,
                  foreground="#666666").pack(fill="x", pady=(0, 8))

    # .. models -----------------------------------------------------------
    def _explorer_label(self):
        d = self.defaults
        b = np.asarray(d["b_vector"], float)
        ang = math.degrees(math.atan2(b[1], b[0]))
        return (f"Explorer x pair: ER {d['er']:.3g}, {d['trap_hz'] / 1e3:.2f} kHz, "
                f"{np.linalg.norm(b):.4f} G at {ang:.0f}°")

    def _refresh_models(self):
        values = [MODEL_PHASE4, MODEL_CAL]
        if self.defaults:
            values.append(self._explorer_label())
        self.model_box.configure(values=values)
        if self.model_var.get() not in values:
            self.model_var.set(MODEL_PHASE4)

    def set_explorer_defaults(self, defaults):
        """New x-pair settings from the sideband explorer (window already open)."""
        explorer_chosen = self.defaults is not None and self.model_var.get() == self._explorer_label()
        self.defaults = defaults
        self._refresh_models()
        if explorer_chosen:
            self.model_var.set(self._explorer_label())
            self._changed()

    def _model_key(self):
        name = self.model_var.get()
        m, h = int(self.m_var.get()), int(self.h_var.get())
        if name == MODEL_CAL:
            return ("calibration", 3, 0)
        if name == MODEL_PHASE4:
            return ("phase4", m, h)
        d = self.defaults
        return ("explorer", m, h, float(d["trap_hz"]), float(d["er"]), float(d["spacing_m"]),
                tuple(np.round(np.asarray(d["b_vector"], float), 7)))

    @staticmethod
    def build_model(key):
        kind = key[0]
        if kind == "calibration":
            return ch.calibration_model()
        if kind == "phase4":
            return ch.first_principles_model(key[1], key[2])
        _, m, h, trap_hz, er, spacing, b = key
        return ch.first_principles_model(m, h, er=er, trap_hz=trap_hz, spacing_m=spacing, b_vector=np.array(b))

    # .. settings <-> widgets ----------------------------------------------
    def settings(self):
        return ch.GateSettings(m=int(self.m_var.get()), n_atoms=int(self.n_var.get()),
                               hold_v=self.r_hold_v.get(), hold_us=self.r_hold.get(),
                               lock_hold=bool(self.lock_var.get()), ramp=self.ramp_var.get(),
                               smooth_ramp_us=self.r_ramp.get(), gap_us=self.r_gap.get(),
                               scattering=self.sc_var.get(), a_a0=self.r_a.get(),
                               transverse_hz=(self.r_fy.get() * 1e3, self.r_fz.get() * 1e3))

    def _pair_chosen(self):
        m = int(self.m_var.get())
        if m != 3 and self.model_var.get() == MODEL_CAL:
            self.model_var.set(MODEL_PHASE4)
        self.r_hold_v.set(DEFAULT_HOLD_V[m])
        self._changed()

    def _reset(self):
        self.n_var.set(5)
        self.m_var.set(3)
        self.model_var.set(MODEL_PHASE4)
        self.h_var.set(int(self.defaults["handedness"]) if self.defaults else -1)
        self.r_hold_v.set(DEFAULT_HOLD_V[3])
        self.ramp_var.set("robust")
        self.r_ramp.set(1000.0)
        self.lock_var.set(True)
        self.r_hold.set(500.0)
        self.r_gap.set(200.0)
        self.sc_var.set("uniform")
        self.r_a.set(ch.DEFAULT_SCATTERING_A0)
        self.r_fy.set(ch.TRANSVERSE_HZ[0] / 1e3)
        self.r_fz.set(ch.TRANSVERSE_HZ[1] / 1e3)
        self._changed()

    def _sync_enabled(self):
        self.r_ramp.enable(self.ramp_var.get() == "smooth")
        self.r_hold.enable(not self.lock_var.get())
        self.r_a.enable(self.sc_var.get() == "uniform")
        if self.model_var.get() == MODEL_CAL and int(self.m_var.get()) != 3:
            self.m_var.set(3)
            self.r_hold_v.set(DEFAULT_HOLD_V[3])

    def _changed(self, *_):
        if self._loading or self.closed:
            return
        self._sync_enabled()
        if self._after is not None:
            self.top.after_cancel(self._after)
        self._after = self.top.after(DEBOUNCE_MS, self._submit)
        self.status.configure(text="computing ...", foreground="#666666")

    # .. background ----------------------------------------------------------
    def _submit(self):
        self._after = None
        if self.closed:
            return
        self._sync_enabled()
        self._gen += 1
        gen = self._gen
        key = self._model_key()
        s = self.settings()
        if key not in self.models:
            if key in self.building:
                return
            self.building.add(key)
            self.status.configure(text="building the packet tracks of the lattice model (about 30 s the first "
                                       "time; cached in data/ afterwards) ...", foreground="#666666")
            t0 = time.perf_counter()

            def build(progress, cancelled, key=key):
                try:
                    return ("ok", key, self.build_model(key), time.perf_counter() - t0)
                except Exception:
                    return ("error", key, traceback.format_exc(), 0.0)
            self.run_background(build, self._built)
            return
        cm = self.models[key]

        def job(progress, cancelled, gen=gen, s=s, cm=cm):
            if gen != self._gen:
                return ("stale", gen, None, 0.0)
            t0 = time.perf_counter()
            try:
                res = ch.compute_sequence(cm, s, hold_scan=False)
                res["model"] = cm
                res["model_key"] = key
                return ("ok", gen, res, time.perf_counter() - t0)
            except Exception:
                return ("error", gen, traceback.format_exc(), 0.0)
        self.run_background(job, self._computed)

    def _built(self, value):
        status, key, model, secs = value
        self.building.discard(key)
        if self.closed:
            return
        if status != "ok":
            print(model, file=sys.stderr)
            self.status.configure(text="building the lattice model failed - see the console", foreground="#c60")
            return
        self.models[key] = model
        self.status.configure(text=f"lattice model ready ({secs:.1f} s)", foreground="#666666")
        self._submit()

    def _computed(self, value):
        status, gen, res, secs = value
        if self.closed or status == "stale" or gen != self._gen:
            return
        if status != "ok":
            print(res, file=sys.stderr)
            self.status.configure(text="calculation failed - see the console", foreground="#c60")
            return
        frac = self.frame / max(1, self.result["frames"]["t_us"].size - 1) if self.result else 0.0
        self.result = res
        self.scan = None
        self.check = None
        nfr = res["frames"]["t_us"].size
        self.frame = int(round(frac * (nfr - 1)))
        self.time_scale.configure(to=nfr - 1)
        self.time_var.set(self.frame)
        if self.lock_var.get():
            self._loading = True
            self.r_hold.set(res["hold_us"])
            self._loading = False
        self.book.mark_dirty()
        self.book.draw_visible()
        self._show_readout()
        self.status.configure(text=f"sequence computed in {secs:.2f} s; hold-voltage scan ...",
                              foreground="#666666")
        s, cm, hold_us, a_m = res["settings"], res["model"], res["hold_us"], res["a_m"]

        def scan_job(progress, cancelled, gen=gen):
            if gen != self._gen:
                return ("stale", gen, None, 0.0)
            t0 = time.perf_counter()
            try:
                sc = ch.hold_voltage_scan(ch.with_transverse(cm, s.transverse_hz), s, hold_us, a_m)
                return ("ok", gen, sc, time.perf_counter() - t0)
            except Exception:
                return ("error", gen, traceback.format_exc(), 0.0)
        self.run_background(scan_job, self._scanned)
        if ch.has_lattice_energies(cm):
            def check_job(progress, cancelled, gen=gen, res=res):
                if gen != self._gen:
                    return ("stale", gen, None, 0.0)
                t0 = time.perf_counter()
                try:
                    return ("ok", gen, ch.seven_state_check(cm, s, res), time.perf_counter() - t0)
                except Exception:
                    return ("error", gen, traceback.format_exc(), 0.0)
            self.run_background(check_job, self._checked)

    def _scanned(self, value):
        status, gen, sc, secs = value
        if self.closed or status == "stale" or gen != self._gen:
            return
        if status != "ok":
            print(sc, file=sys.stderr)
            self.status.configure(text="hold-voltage scan failed - see the console", foreground="#c60")
            return
        self.scan = sc
        self.p_ram.mark_dirty()
        self.p_ram.draw_if_visible()
        self.status.configure(text=f"hold-voltage scan in {secs:.2f} s", foreground="#666666")

    def _checked(self, value):
        status, gen, chk, secs = value
        if self.closed or status == "stale" or gen != self._gen:
            return
        if status == "ok":
            self.check = dict(chk, seconds=secs)
            self._show_readout()

    # .. drawing -------------------------------------------------------------
    def _draw_sequence(self, fig):
        if self.result is None:
            self.labplot.placeholder(fig, "computing ...")
            self.art = None
            return
        self.art = draw_sequence(fig, self.result, self.labplot)
        self._show_time(update_frame(self.art, self.result, self.frame))

    def _show_time(self, lines):
        line_t, line_z = lines
        self.time_label.configure(text=line_t + "      meeting point: " + line_z.replace("\n", ", "))

    def _draw_ramsey(self, fig):
        if self.result is None:
            self.labplot.placeholder(fig, "computing ...")
            return
        draw_ramsey(fig, self.result, self.scan, self.labplot)

    def _on_draw(self, event):
        """After every full draw: keep the static background, then draw the moving artists."""
        if self._saving or self.art is None:
            self._bg = None
            return
        canvas = self.p_seq.canvas
        self._bg = canvas.copy_from_bbox(self.p_seq.fig.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for a in self.art["animated"]:
            if a.get_visible():
                a.axes.draw_artist(a)

    def _wrap_save(self, panel):
        """Saving draws the animated artists too (they are skipped by a normal draw)."""
        def save(path=None, dpi=self.labplot.SAVE_DPI):
            arts = self.art["animated"] if self.art else []
            self._saving = True
            for a in arts:
                a.set_animated(False)
            if self.art is not None:
                update_frame(self.art, self.result, self.frame, texts=True)
            try:
                return self.labplot.PlotPanel.save_figure(panel, path, dpi)
            finally:
                for a in arts:
                    a.set_animated(True)
                if self.art is not None:
                    update_frame(self.art, self.result, self.frame)
                self._saving = False
                panel.canvas.draw()
        panel.save_figure = save

    def show_frame(self, i):
        if self.result is None:
            return
        n = self.result["frames"]["t_us"].size
        self.frame = int(min(max(i, 0), n - 1))
        if self.art is None or self.p_seq.dirty or not self.p_seq.is_visible():
            return
        self._show_time(update_frame(self.art, self.result, self.frame))
        canvas = self.p_seq.canvas
        if self._bg is None:
            canvas.draw_idle()
            return
        canvas.restore_region(self._bg)
        self._draw_animated()
        canvas.blit(self.p_seq.fig.bbox)

    def _slid(self, _=None):
        self.show_frame(int(round(self.time_var.get())))
        self._acc = float(self.frame)

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            if self.result is not None and self.frame >= self.result["frames"]["t_us"].size - 1:
                self.frame = 0
            self._acc = float(self.frame)
            self._last_tick = time.perf_counter()
            self._tick()
        elif self._tick_after is not None:
            self.top.after_cancel(self._tick_after)
            self._tick_after = None

    def _tick(self):
        self._tick_after = None
        if not self.playing or self.closed:
            return
        if self.result is not None:
            n = self.result["frames"]["t_us"].size
            now = time.perf_counter()
            # advance by the wall time since the last tick, so the playback length holds when drawing is slow
            step = (n - 1) * (now - self._last_tick) / PLAYBACK_S.get(self.speed_var.get(), 10.0)
            self._last_tick = now
            self._acc += step
            if self._acc > n - 1:
                self._acc = 0.0
            self.time_var.set(int(self._acc))
            self.show_frame(int(self._acc))
        self._tick_after = self.top.after(TICK_MS, self._tick)

    def _show_readout(self):
        res = self.result
        s = res["settings"]
        hold = res["hold"]
        phi_hold = res["phi1"] - res["phi_ramp"] if res["hold_us"] > 0 else 0.0
        a_a0 = res["a_m"] / A0
        cap = " (capped)" if res["hold_capped"] else ""
        lines = [
            f"Leg: ramp out {res['ramp_us'] / 1e3:.3f} ms, hold {res['hold_us']:.1f} µs{cap}, ramp back "
            f"{res['ramp_us'] / 1e3:.3f} ms; whole sequence {res['t_end_us'] / 1e3:.3f} ms.   "
            f"At the hold ({s.hold_v:.0f} V, θ = {np.degrees(res['theta_hold']):.2f}°): separation "
            f"{hold['separation_m'] * 1e9:.1f} nm, packet widths {hold['widths_m'][0] * 1e9:.1f} / "
            f"{hold['widths_m'][1] * 1e9:.1f} nm, overlap / overlap at V_π {hold['rel_overlap']:.4g}, "
            f"U/h {hold['u_hz'] / 1e3:.4g} kHz (a = {a_a0:.0f} a0, transverse {s.transverse_hz[0] / 1e3:.2f} / "
            f"{s.transverse_hz[1] / 1e3:.2f} kHz).",
            f"Collisional phase per neighbour pair: leg 1 {res['phi1'] / np.pi:.4f} π (ramps "
            f"{res['phi_ramp'] / np.pi:.4f} π, hold {phi_hold / np.pi:.4f} π), leg 2 {res['phi2'] / np.pi:.4f} π; "
            f"total Φ = {(res['phi1'] + res['phi2']) / np.pi:.4f} π (Φ = π: √ZZ, equivalent to CZ up to "
            f"single-atom phases).",
        ]
        if not ch.has_lattice_energies(res["model"]):
            lines.append("7-state model (collision.py evolve): not available for the calibration files "
                         "(no lattice energies).")
        elif self.check is None:
            lines.append("7-state model (collision.py evolve, leg 1): computing ...")
        else:
            c = self.check
            lines.append(f"7-state model (collision.py evolve, leg 1, {'DIS' if s.scattering == 'DIS' else 'uniform'}"
                         f" lengths): phase {c['phi'] / np.pi:.4f} π, leakage out of |L:+{s.m}, R:−{s.m}⟩ "
                         f"{c['leakage']:.1e} ({c['seconds']:.1f} s).")
        self.summary.configure(text="\n".join(lines))
        self.tree.delete(*self.tree.get_children())
        sim = res["fringes_sim"]
        scan = res["scan_phases"]
        for k in range(s.n_atoms):
            c_rule, shift = contrast_and_shift(res["factors"][k])
            # the simulated fringe's first Fourier component: 1 - 2P = Re[F e^{i phase}]
            c_sim, _ = contrast_and_shift(2 * np.mean((1 - 2 * sim[k][:-1]) * np.exp(-1j * scan[:-1])))
            self.tree.insert("", "end", values=(f"atom {k + 1}", int(res["neighbours"][k]), f"{c_rule:+.4f}",
                                                f"{c_sim:+.4f}", f"{shift:+.2f}"))
        self.tree.configure(height=s.n_atoms)

    # .. lifetime ------------------------------------------------------------
    def lift(self):
        self.top.deiconify()
        self.top.lift()

    def alive(self):
        if self.closed:
            return False
        try:
            return bool(self.top.winfo_exists())
        except Exception:
            return False

    def close(self):
        self.closed = True
        self.playing = False
        for a in (self._after, self._tick_after):
            if a is not None:
                try:
                    self.top.after_cancel(a)
                except Exception:
                    pass
        if self.worker is not None:
            self.worker.close()
        self.top.destroy()


def main():
    win = ChainWindow()
    win.top.mainloop()


if __name__ == "__main__":
    import os as _os
    _os.chdir(REPO)
    main()
