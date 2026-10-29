"""Molasses window: temperature, scattering, hopping and field sensitivity of
the imaging / PGC molasses in the full-depth lattice (lattice_sim.molasses).

Two sources of numbers:

* tables - the Monte Carlo scans of ``scripts/molasses_scan.py``
  (data/molasses/intensity.jsonl, field.jsonl), interpolated at the chosen
  detuning and intensity (``lattice_sim.molasses.tables``; full lattice depth,
  zero field, no extrapolation outside the grid);
* live - a Monte Carlo run at exactly the chosen settings (field, depth
  scale and duration included) on a worker thread; ~10-60 s for 64 atoms,
  longer close to resonance and at high intensity. "Field scan" runs the
  three axes at 0-400 mG (13 runs).

Tabs: Versus intensity (T, scattering rate, hopping rate, scattering rate
against T), Versus field (T and site survival against B along x, y, z;
the field at which T doubles), Live run (energy distributions per axis).

Survival assumes every hop loses the atom from its site (tables.HOP_IS_LOSS)
and uses the pulse duration set on the left. Axes x, y, z are the lattice
axes (model.Config); the field acts on the ground state only.

Usage: python scripts/molasses_explorer.py      # standalone window
       (or the "Molasses" group of scripts/sideband_explorer.py)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lattice_sim.constants import KB  # noqa: E402
from lattice_sim.molasses import model as M  # noqa: E402
from lattice_sim.molasses import tables as T  # noqa: E402

AXES = ("x", "y", "z")
AXIS_COLOURS = ("#0072B2", "#D55E00", "#009E73")          # Okabe-Ito
CURRENT_COLOUR = "#000000"
LIVE_FIELDS_MG = (0.0, 50.0, 100.0, 200.0, 400.0)
ATTEMPT_CHOICES = {"1/τ_E (model)": None, f"{T.LI_ATTEMPT_RATE_PER_S:g} s⁻¹ (Li 2009 fit)": T.LI_ATTEMPT_RATE_PER_S}
DEFAULTS = dict(detuning=-51.0, intensity=0.7, pulse_ms=250.0, attempt="1/τ_E (model)", bx=0.0, by=0.0, bz=0.0, depth_scale=1.0,
                live_ms=2.0, trap_khz=M.TRAP_FULL_KHZ)


def attempt_of(s):
    return ATTEMPT_CHOICES.get(s.get("attempt"), None)


def detuning_colours(n):
    import matplotlib as mpl
    return [mpl.colormaps["viridis"](v) for v in np.linspace(0.05, 0.85, max(n, 1))][::-1]   # as figure_molasses


# ------------------------------------------------------------------ live runs
def live_config(s, b_mg=None, n_atoms=64):
    burn = min(3.0, 0.6 + 0.1 / max(s["intensity"], 1e-3))
    return M.Config(intensity=s["intensity"], detuning=s["detuning"],
                    b_mg=tuple(b_mg if b_mg is not None else (s["bx"], s["by"], s["bz"])),
                    depth_scale=s["depth_scale"], trap_khz=tuple(s["trap_khz"]), n_atoms=n_atoms,
                    t_total_ms=burn + s["live_ms"], t_burn_ms=burn, seed=7)


def live_point(s, n_atoms):
    r = M.run(live_config(s, n_atoms=n_atoms))
    att = attempt_of(s)
    hop = r.hop_rate() if att is None else M.arrhenius_hop_rate(r.t_uk, 1e3 / att, r.barrier_uk())
    return dict(settings=dict(s), t=r.t_uk, t_err=r.t_uk_err, scatter=r.scatter_rate, depump=r.depump_frac,
                tau=r.tau_e_ms, light_shift=r.light_shift_uk, hop=hop, hops_obs=r.hops_per_s,
                barrier=r.barrier_uk(), e=r.e_samples)


def live_field_scan(s, report, cancelled, n_atoms=64):
    jobs = [(0, 0.0)] + [(ax, b) for ax in range(3) for b in LIVE_FIELDS_MG[1:]]
    out = []
    for k, (ax, b) in enumerate(jobs):
        if cancelled():
            break
        bv = [0.0, 0.0, 0.0]
        bv[ax] = b
        r = M.run(live_config(s, b_mg=bv, n_atoms=n_atoms))
        att = attempt_of(s)
        hop = r.hop_rate() if att is None else M.arrhenius_hop_rate(r.t_uk, 1e3 / att, r.barrier_uk())
        out.append((ax, b, r.t_uk, hop))
        report(k + 1, len(jobs))
    return dict(settings=dict(s), points=out)


def field_curves_from_live(scan):
    """axis -> (B, T (n,3), hop (n,3)), the zero-field point shared."""
    zero = [p for p in scan["points"] if p[1] == 0.0]
    curves = {}
    for ax in range(3):
        pts = zero + [p for p in scan["points"] if p[0] == ax and p[1] != 0.0]
        pts.sort(key=lambda p: p[1])
        curves[ax] = (np.array([p[1] for p in pts]), np.array([p[2] for p in pts]),
                      np.array([p[3] for p in pts]))
    return curves


# ------------------------------------------------------------------ drawing
def draw_intensity(fig, tab, s, live, labplot):
    fig.clear()
    if tab is None or not np.isfinite(tab.scatter).any():
        labplot.placeholder(fig, "no table yet: run  python scripts/molasses_scan.py intensity")
        return
    ax = fig.subplots(2, 2)
    cols = detuning_colours(len(tab.detunings))
    att = attempt_of(s)
    hop_nodes = tab.node_hop_rates(att).sum(axis=-1)
    ii = np.geomspace(tab.intensities[0], tab.intensities[-1], 60)
    for k, d in enumerate(tab.detunings):
        lab = f"{d:+.0f} MHz"
        ok = np.isfinite(tab.scatter[k])
        xi = tab.intensities[ok]
        tm = tab.t[k][ok].mean(axis=1)
        te = np.sqrt((tab.t_err[k][ok] ** 2).sum(axis=1)) / 3
        ax[0, 0].errorbar(xi, tm, te, color=cols[k], marker="o", ms=3, lw=1, label=lab)
        ax[0, 1].plot(xi, tab.scatter[k][ok] * 1e-3, color=cols[k], marker="o", ms=3, lw=1)
        ax[1, 0].plot(xi, np.maximum(hop_nodes[k][ok], 1e-12), color=cols[k], marker="o", ms=3, lw=1)
        ax[1, 1].plot(tm, tab.scatter[k][ok] * 1e-3, color=cols[k], marker="o", ms=3, lw=1)
    d, i = s["detuning"], s["intensity"]
    tc = np.array([tab.temperature(d, x).mean() for x in ii])
    rc = np.array([tab.scattering(d, x) for x in ii]) * 1e-3
    hc = np.array([tab.hop_rate(d, x, att).sum() for x in ii])
    kw = dict(color=CURRENT_COLOUR, ls="--", lw=1.2, label=f"{d:+.1f} MHz (interpolated)")
    ax[0, 0].plot(ii, tc, **kw)
    ax[0, 1].plot(ii, rc, **kw)
    ax[1, 0].plot(ii, np.maximum(hc, 1e-12), **kw)
    ax[1, 1].plot(tc, rc, **kw)
    t0, r0, h0 = tab.temperature(d, i).mean(), tab.scattering(d, i) * 1e-3, tab.hop_rate(d, i, att).sum()
    for a, (x, y) in zip(ax.flat, ((i, t0), (i, r0), (i, max(h0, 1e-12)), (t0, r0))):
        a.plot([x], [y], marker="*", ms=11, color=CURRENT_COLOUR, ls="none")
    if live is not None and live["settings"]["detuning"] == d and live["settings"]["intensity"] == i:
        tl = live["t"].mean()
        ax[0, 0].errorbar([i], [tl], [np.sqrt((live["t_err"] ** 2).sum()) / 3], color="#CC79A7", marker="s",
                          ms=5, ls="none", label="live run")
        ax[1, 1].plot([tl], [live["scatter"] * 1e-3], color="#CC79A7", marker="s", ms=5, ls="none")
    for a in ax.flat[:3]:
        a.set_xscale("log")
    ax[1, 0].set_yscale("log")
    labplot.finish(ax[0, 0], "Intensity per beam (mW cm$^{-2}$)", "Temperature (µK)")
    labplot.finish(ax[0, 1], "Intensity per beam (mW cm$^{-2}$)", "Scattering rate (10$^3$ s$^{-1}$)")
    labplot.finish(ax[1, 0], "Intensity per beam (mW cm$^{-2}$)", "Hopping rate, x+y+z (s$^{-1}$)")
    labplot.finish(ax[1, 1], "Temperature (µK)", "Scattering rate (10$^3$ s$^{-1}$)")
    ax[1, 0].set_ylim(bottom=max(1e-6, ax[1, 0].get_ylim()[0]))
    labplot.legend(ax[0, 0], loc="upper center", fontsize="small")
    for a, letter in zip(ax.flat, "abcd"):
        a.text(-0.14, 1.02, letter, transform=a.transAxes, fontweight="bold", va="bottom")
    labplot.note(ax[1, 0], f"Arrhenius, attempt rate {s.get('attempt')}; full depth", "bottom-right")


def draw_field(fig, tab, s, scan, labplot):
    fig.clear()
    ax = fig.subplots(1, 3)
    pulse = s["pulse_ms"]
    drew = False
    pair = tab.nearest_field_pair(s["detuning"], s["intensity"]) if tab is not None else None
    if pair is not None:
        for ka in range(3):
            c = tab.field_curve(pair[0], pair[1], ka)
            if c is None:
                continue
            b, t, tau, bar = c
            att = attempt_of(s)
            hop = M.arrhenius_hop_rate(t, tau if att is None else 1e3 / att, bar[None, :])
            ax[0].plot(b, t.mean(axis=1), color=AXIS_COLOURS[ka], marker="o", ms=3, lw=1,
                       label=f"B ∥ {AXES[ka]} (table {pair[0]:+.0f} MHz, {pair[1]:g} mW cm$^{{-2}}$)")
            ax[1].plot(b, T.survival(hop, pulse), color=AXIS_COLOURS[ka], marker="o", ms=3, lw=1)
            drew = True
    if scan is not None:
        for ka, (b, t, hop) in field_curves_from_live(scan).items():
            if len(b) < 2:
                continue
            ax[0].plot(b, t.mean(axis=1), color=AXIS_COLOURS[ka], marker="s", ms=4, ls="--", lw=1.2,
                       label=f"B ∥ {AXES[ka]} (live)")
            ax[1].plot(b, T.survival(hop, pulse), color=AXIS_COLOURS[ka], marker="s", ms=4, ls="--", lw=1.2)
            drew = True
    if tab is not None and tab.field_rows:
        cols = detuning_colours(len(tab.detunings))
        for k, d in enumerate(tab.detunings):
            ints = sorted({p[1] for p in tab.field_pairs() if p[0] == d})
            if not ints:
                continue
            b2 = [np.mean([tab.doubling_field(d, i, a) for a in range(3)]) for i in ints]
            ax[2].plot(ints, b2, color=cols[k], marker="o", ms=4, lw=1, label=f"{d:+.0f} MHz")
        labplot.legend(ax[2], fontsize="small")
    if not drew:
        labplot.placeholder(fig, "no field data: run  python scripts/molasses_scan.py field  or 'Field scan'")
        return
    labplot.finish(ax[0], "Field along the axis (mG)", "Temperature (µK)")
    labplot.finish(ax[1], "Field along the axis (mG)", f"Site survival after {pulse:g} ms")
    labplot.finish(ax[2], "Intensity per beam (mW cm$^{-2}$)", "Field doubling T, mean of x, y, z (mG)")
    ax[1].set_ylim(-0.02, 1.02)
    labplot.legend(ax[0], fontsize="small")
    for a, letter in zip(ax, "abc"):
        a.text(-0.14, 1.02, letter, transform=a.transAxes, fontweight="bold", va="bottom")


def draw_live(fig, live, labplot):
    fig.clear()
    if live is None:
        labplot.placeholder(fig, "press 'Run Monte Carlo at these settings'")
        return
    ax = fig.subplots(1, 3, sharey=True)
    e_uk = live["e"] / KB * 1e6                   # (N, ns, 3), lattice energy per axis
    for k in range(3):
        e = e_uk[:, :, k].ravel()
        hi = max(np.percentile(e, 99.5), 1.0)
        ax[k].hist(e, bins=np.linspace(0, hi, 50), density=True, color=AXIS_COLOURS[k], alpha=0.6,
                   label="Monte Carlo")
        x = np.linspace(0, hi, 200)
        t = live["t"][k]
        ax[k].plot(x, np.exp(-x / t) / t, color="k", lw=1, label=f"Boltzmann, T = {t:.1f} µK")
        ax[k].set_yscale("log")
        labplot.finish(ax[k], f"Energy along {AXES[k]} (µK)", "Probability density (µK$^{-1}$)" if k == 0 else None)
        ax[k].legend(fontsize="small")
        ax[k].text(-0.1, 1.02, "abc"[k], transform=ax[k].transAxes, fontweight="bold", va="bottom")
    s = live["settings"]
    labplot.note(ax[0], f"{s['detuning']:+.1f} MHz, {s['intensity']:g} mW cm$^{{-2}}$, "
                        f"B = ({s['bx']:g}, {s['by']:g}, {s['bz']:g}) mG, depth ×{s['depth_scale']:g}", "top-right")


# ------------------------------------------------------------------ window
class MolassesWindow:
    """The molasses window: a Toplevel of ``master`` (or its own Tk root).
    defaults: overrides of DEFAULTS (e.g. trap_khz from the sideband explorer)."""

    def __init__(self, master=None, defaults=None):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk
        import labplot
        from chain_explorer import Worker
        self.tk, self.ttk, self.labplot = tk, ttk, labplot
        if master is None:
            self.top = tk.Tk()
            labplot.apply_style()
        else:
            self.top = tk.Toplevel(master)
        self.top.geometry("1480x920")
        self.top.title("Molasses: temperature, hopping and field sensitivity - lattice-sim")
        self.closed = False
        self.s = dict(DEFAULTS)
        self.s.update(defaults or {})
        self.tab = None
        self.live = None
        self.scan = None
        self.job = None
        self.worker = Worker(self.top, on_error=lambda text: self._status(text, "#c60"))
        base = tkfont.nametofont("TkDefaultFont")
        self.small = base.copy()
        self.small.configure(size=max(7, base.cget("size") - 1))

        outer = ttk.Panedwindow(self.top, orient="horizontal")
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, padding=(8, 4), width=340)
        right = ttk.Frame(outer)
        outer.add(left, weight=0)
        outer.add(right, weight=1)
        self._build_settings(left)

        self.readout = ttk.Label(right, text="", justify="left", padding=(6, 4), wraplength=1080)
        self.readout.pack(side="bottom", fill="x")
        nb = ttk.Notebook(right)
        nb.pack(side="top", fill="both", expand=True)
        self.book = labplot.PlotBook(nb)
        figs = str(REPO / "figures")
        self.book.add("intensity", "Versus intensity", figsize=(10, 7), save_dir=figs,
                      save_name=lambda: f"molasses_vs_intensity_{self.s['detuning']:+.0f}MHz",
                      draw=lambda fig: draw_intensity(fig, self.tab, self.s, self.live, labplot))
        self.book.add("field", "Versus field", figsize=(12, 4.6), save_dir=figs,
                      save_name=lambda: f"molasses_vs_field_{self.s['detuning']:+.0f}MHz",
                      draw=lambda fig: draw_field(fig, self.tab, self.s, self.scan, labplot))
        self.book.add("live", "Live run", figsize=(12, 4.6), save_dir=figs, save_name="molasses_live",
                      draw=lambda fig: draw_live(fig, self.live, labplot))
        self.status = ttk.Label(self.top, text="", foreground="#666666", anchor="w", padding=(6, 2))
        self.status.pack(fill="x", side="bottom")
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.reload()

    # .. settings -----------------------------------------------------------
    def _build_settings(self, parent):
        ttk = self.ttk
        self.rows = {}

        def group(title):
            g = ttk.LabelFrame(parent, text=title, padding=6)
            g.pack(fill="x", pady=(6, 0))
            return g

        g = group("Molasses")
        self._row(g, "detuning", "Detuning from F=4→F'=5", "MHz", -60.0, -5.0, 1,
                  "Negative = red. Tables: -51 to -10 MHz, total saturation ≤ 0.1.")
        self._row(g, "intensity", "Intensity per beam", "mW/cm²", 0.05, 3.0, 3,
                  "At the atoms, each of the six beams (retro included). Tables: 0.05-3.")
        self._row(g, "pulse_ms", "Pulse duration", "ms", 0.1, 5000.0, 1,
                  "Duration of the molasses in the survival measurement (survival = exp(-hops × t)).")
        self.attempt = self.tk.StringVar(value=self.s["attempt"])
        f = ttk.Frame(g)
        f.pack(fill="x", pady=(5, 0))
        ttk.Label(f, text="Hopping attempt rate").pack(side="left")
        box = ttk.Combobox(f, textvariable=self.attempt, values=list(ATTEMPT_CHOICES), width=20, state="readonly")
        box.pack(side="right")

        def picked(_=None):
            self.s["attempt"] = self.attempt.get()
            self._changed("attempt")
        box.bind("<<ComboboxSelected>>", picked)
        ttk.Label(g, text="Γ_h = Γ_a erfc(√(U/k_B T)) per axis. The model's 1/τ_E is ~10× Li's fitted Γ_a.",
                  font=self.small, foreground="#666666", wraplength=300, justify="left").pack(fill="x")
        g = group("Live Monte Carlo only")
        for key, name in (("bx", "Field along x"), ("by", "Field along y"), ("bz", "Field along z")):
            self._row(g, key, name, "mG", -800.0, 800.0, 0, "")
        self._row(g, "depth_scale", "Lattice depth (fraction of full)", "", 0.05, 1.0, 2,
                  f"Full depth: {', '.join(f'{f:g}' for f in self.s['trap_khz'])} kHz trap frequencies (x, y, z).")
        self._row(g, "live_ms", "Recorded time per run", "ms", 0.5, 10.0, 1,
                  "After an automatic burn-in. Longer = smaller error bars, proportionally slower.")
        self.atoms = self.tk.StringVar(value="64")
        f = ttk.Frame(g)
        f.pack(fill="x", pady=(5, 0))
        ttk.Label(f, text="Atoms").pack(side="left")
        ttk.Combobox(f, textvariable=self.atoms, values=("32", "64", "128", "256"), width=6,
                     state="readonly").pack(side="right")
        b = ttk.Frame(parent)
        b.pack(fill="x", pady=(8, 0))
        ttk.Button(b, text="Run Monte Carlo at these settings", command=self.run_live).pack(fill="x")
        ttk.Button(b, text="Field scan x, y, z (0-400 mG, 13 runs)", command=self.run_scan).pack(fill="x", pady=2)
        ttk.Button(b, text="Cancel field scan", command=self.cancel).pack(fill="x")
        ttk.Button(b, text="Reload tables", command=self.reload).pack(fill="x", pady=(8, 0))
        lbl = ttk.Label(parent, font=self.small, foreground="#666666", wraplength=320, justify="left",
                        text="Semiclassical Monte Carlo wave-function model of the six-beam lin⊥lin molasses "
                             "in the lattice (Cs F=4, F'=5,4,3, depumping and repumping). Assumed: beam azimuths, "
                             "molasses phases random per atom, hop = loss. See docs/findings.md (2026-10-29).")
        lbl.pack(fill="x", pady=(10, 0))

    def _row(self, parent, key, label, unit, lo, hi, digits, help_text):
        tk, ttk = self.tk, self.ttk
        text = tk.StringVar(value=f"{self.s[key]:.{digits}f}")
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(4, 0))
        ttk.Label(frame, text=label).pack(side="left")
        ttk.Label(frame, text=unit, width=7).pack(side="right")
        entry = ttk.Entry(frame, textvariable=text, width=8, justify="right")
        entry.pack(side="right")

        def typed(_=None):
            try:
                v = float(text.get())
            except ValueError:
                text.set(f"{self.s[key]:.{digits}f}")
                return
            v = min(max(v, lo), hi)
            text.set(f"{v:.{digits}f}")
            if v != self.s[key]:
                self.s[key] = v
                self._changed(key)
        entry.bind("<Return>", typed)
        entry.bind("<FocusOut>", typed)
        if help_text:
            ttk.Label(parent, text=help_text, font=self.small, foreground="#666666", wraplength=300,
                      justify="left").pack(fill="x")
        self.rows[key] = text

    def _changed(self, key):
        if key in ("detuning", "intensity", "pulse_ms", "attempt"):
            self.book.mark_dirty()
            self.book.draw_visible()
            self._readout()

    # .. data ---------------------------------------------------------------
    def reload(self):
        self.tab = T.Tables.from_dir()
        n = 0 if self.tab is None else int(np.isfinite(self.tab.scatter).sum())
        self._status(f"tables: {n} zero-field points ({len(self.tab.detunings)} detunings × "
                     f"{len(self.tab.intensities)} intensities), {len(self.tab.field_rows)} field points "
                     f"from data/molasses/")
        self.book.mark_dirty()
        self.book.draw_visible()
        self._readout()

    def _readout(self):
        s, tab = self.s, self.tab
        if tab is None or not np.isfinite(tab.scatter).any():
            self.readout.configure(text="no tables - run scripts/molasses_scan.py")
            return
        d, i = s["detuning"], s["intensity"]
        t = tab.temperature(d, i)
        h = tab.hop_rate(d, i, attempt_of(s))
        r = tab.scattering(d, i)
        flag = "" if tab.inside(d, i) else "   (outside the table: clamped to its edge)"
        sat = M.total_saturation(i, d)
        if sat > M.MAX_SATURATION:
            flag += f"   (saturation s = {sat:.2f} > {M.MAX_SATURATION:g}: outside the model, not tabulated)"
        elif not tab.complete:
            flag += "   (table incomplete: nan next to missing points)"
        txt = (f"Table at {d:+.1f} MHz, {i:g} mW/cm² per beam{flag}:  T = {t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} µK "
               f"(x / y / z);  scattering {r:.3g} s⁻¹ ({r / t.mean():.3g} s⁻¹ µK⁻¹);  "
               f"hopping {h.sum():.2g} s⁻¹;  survival after {s['pulse_ms']:g} ms: "
               f"{T.survival(h, s['pulse_ms'])[0]:.4f}")
        if self.live is not None:
            L = self.live
            ls = L["settings"]
            txt += (f"\nLive at {ls['detuning']:+.1f} MHz, {ls['intensity']:g} mW/cm², "
                    f"B = ({ls['bx']:g}, {ls['by']:g}, {ls['bz']:g}) mG, depth ×{ls['depth_scale']:g}:  "
                    f"T = {' / '.join(f'{a:.2f}±{b:.2f}' for a, b in zip(L['t'], L['t_err']))} µK;  "
                    f"scattering {L['scatter']:.3g} s⁻¹;  τ_E = {L['tau'].mean() * 1e3:.0f} µs;  "
                    f"light shift {L['light_shift']:.1f} µK;  F=3 per photon {L['depump']:.2g};  "
                    f"hopping {L['hop'].sum():.2g} s⁻¹ (observed {L['hops_obs'].sum():.2g} s⁻¹)")
        self.readout.configure(text=txt)

    # .. jobs ---------------------------------------------------------------
    def _status(self, text, colour="#666666"):
        try:
            self.status.configure(text=text, foreground=colour)
        except Exception:
            pass

    def run_live(self):
        s = dict(self.s)
        n = int(self.atoms.get())
        sat = M.total_saturation(s["intensity"], s["detuning"])
        warn = f"  WARNING: saturation {sat:.2f} > {M.MAX_SATURATION:g}, outside the model" \
            if sat > M.MAX_SATURATION else ""
        self._status(f"Monte Carlo: {n} atoms at {s['detuning']:+.1f} MHz, {s['intensity']:g} mW/cm² ...{warn}")

        def done(res):
            if self.closed:
                return
            self.live = res
            self._status("Monte Carlo done")
            self.book.mark_dirty()
            self.book.draw_visible()
            self._readout()
        self.worker.run_background(lambda report, cancelled: live_point(s, n), done)

    def run_scan(self):
        s = dict(self.s)
        n = int(self.atoms.get())
        self._status("field scan started ...")

        def done(res):
            if self.closed:
                return
            self.scan = res
            self._status(f"field scan done ({len(res['points'])} runs)")
            self.book.mark_dirty(["field"])
            self.book.draw_visible()

        def progress(i, total):
            self._status(f"field scan: {i} / {total} runs")
        self.job = self.worker.run_background(lambda report, cancelled: live_field_scan(s, report, cancelled, n),
                                              done, progress)

    def cancel(self):
        if self.job is not None:
            self.job.cancel()
            self._status("field scan: cancelling after the current run")

    # .. lifetime -------------------------------------------------------------
    def set_explorer_defaults(self, defaults):
        self.s.update(defaults or {})

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
        self.worker.close()
        self.top.destroy()


def main():
    win = MolassesWindow()
    win.top.mainloop()


if __name__ == "__main__":
    import os as _os
    _os.chdir(REPO)
    main()
