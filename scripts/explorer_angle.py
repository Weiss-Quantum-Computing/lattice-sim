"""Sideband-explorer extension: the EOM arm's polarization angle over the
full range -180 ... +180 deg, and the zero-point energy versus rotation.

As the angle grows, |4,−4⟩ and |3,−3⟩ wells move apart in opposite
directions (spin-dependent transport), by exactly half a lattice spacing
each at ±180 deg, where the |3,−3⟩ well of the neighbouring site sits on
the |4,−4⟩ well the atom started in (``lattice_sim.spectra.sidebands``,
"Large rotation angles"). This module adds:

* an "Energies vs angle" tab for the edited pair: potential minimum V_min
  (the light shift at the trap minimum), ground level E_0, zero-point
  energy E_0 − V_min and level spacing of both states, and the pair's part
  of the carrier, E_0(|4,−4⟩) − E_0(|3,−3⟩), and its zero-point part, over
  −180 ... 180 deg (``sidebands.energies_vs_angle``). Computed on the
  worker thread in short chunks (the spectrum keeps updating), cached by
  the pair's settings (all but the angle) and the field; a dashed line marks
  the current angle;
* an energy-reference option in the Wells tab: each state drawn from its own
  minimum (the shapes), or both on one axis from the free atom at the bias
  field, so V_min and the zero-point energy show; at large angles the tab
  shows the |3,−3⟩ well that the atom's |4,−4⟩ well overlaps (its own site's
  or the neighbouring site's);
* table rows: the site of the nearest |3,−3⟩ well and the distance to it,
  the fraction of the line strength in the carrier and the Δn = ±1 lines,
  and the centroid of the whole line strength (sum rule).
"""

import time
import traceback

import numpy as np

from lattice_sim.spectra.sidebands import energies_vs_angle

ANGLE_STEPS_DEG = (2.5, 5.0, 10.0, 15.0)   # choices for the angle grid of the energies tab
DEFAULT_STEP_DEG = 5.0
CHUNK_ANGLES = 10                          # angles per worker job (~0.4 s), so spectrum updates interleave
CACHE_SIZE = 16
REFERENCES = ("each state from its own minimum", "common zero: free atom at the bias field")
WINDOW_MARGIN_NM = 380.0                   # Wells tab: shown beyond the outermost well centre
LEVELS_SHOWN = 6
MINUS = "−"


def state_label(state):
    f, m = state
    return f"|{f},{m}⟩".replace("-", MINUS)


# ------------------------------------------------------------------ model
def energies_key(settings):
    """What the energies-vs-angle curves depend on: the edited pair's settings
    except the angle, and the bias field."""
    ax = settings["edited"]
    pair = tuple(sorted((k, v) for k, v in settings["pairs"][ax].items() if k != "theta_deg"))
    g = settings["glob"]
    return ax, pair, g["b_gauss"], g["b_angle_deg"]


def angle_grid(step_deg):
    n = int(round(360.0 / step_deg))
    return np.linspace(-180.0, 180.0, n + 1)


def energies_for(se, settings, thetas_deg, progress=None, cancelled=None):
    """``energies_vs_angle`` of the edited pair at the explorer settings."""
    site = se.site_from(settings)
    cfg = next(c for c in site.axes if c.axis == settings["edited"])
    return energies_vs_angle(cfg, site.b_lab_gauss, np.deg2rad(thetas_deg), progress=progress,
                             cancelled=cancelled)


def merge_chunks(chunks, thetas_deg):
    """One energies result from chunk results, in angle order."""
    first = chunks[0]
    states = list(first["states"])
    out = dict(theta=np.deg2rad(np.asarray(thetas_deg, float)), axis=first["axis"],
               states={s: {k: np.concatenate([c["states"][s][k] for c in chunks]) for k in first["states"][s]}
                       for s in states})
    for k in ("carrier_hz", "v_min_diff_hz", "zpe_diff_hz"):
        if k in first:
            out[k] = np.concatenate([c[k] for c in chunks])
    return out


def shown_lower_site(axis_data, lower):
    """Site k of the final-state well drawn in the Wells tab: the solved well
    with the largest diagonal (carrier) weight, else the nearest solved one."""
    sites = axis_data["levels"][lower].get("sites", {0: None})
    weights = axis_data.get("diagonal_sites", {0: 0.0})
    best = max(sites, key=lambda k: weights.get(k, 0.0))
    if weights.get(best, 0.0) > 0:
        return best
    x_u = axis_data["levels"][next(iter(axis_data["levels"]))]["x0_m"]
    return min(sites, key=lambda k: abs(axis_data["levels"][lower]["sites"][k]["x0_m"] - x_u))


# ------------------------------------------------------------------ drawing
def draw_energies(fig, se, res, current_deg, settings_text, waiting="Computing the energies over −180 … 180 deg."):
    fig.clear()
    if res is None:
        fig.text(0.5, 0.5, waiting, ha="center", va="center", color="#666666")
        return
    axes = fig.subplots(2, 3)
    deg = np.rad2deg(res["theta"])
    states = list(res["states"])
    pair = res["axis"]
    xlabel = f"{pair}-pair polarization angle (deg)"
    panels = [
        (axes[0, 0], "v_min_hz", "Potential minimum V_min (kHz)", "Light shift at the trap minimum"),
        (axes[0, 1], "e0_hz", "Ground level E_0 (kHz)", "Vibrational ground level"),
        (axes[0, 2], "zpe_hz", "E_0 − V_min (kHz)", "Zero-point energy"),
        (axes[1, 0], "trap_hz", "E_1 − E_0 (kHz)", "Vibrational level spacing"),
    ]
    for ax, key, ylabel, title in panels:
        for s in states:
            ax.plot(deg, res["states"][s][key] / 1e3, color=se.STATE_COLOURS.get(s, "#444444"), lw=1.2,
                    marker="o", ms=2.0, label=state_label(s))
        se._finish(ax, xlabel, ylabel, title)
    axes[0, 0].legend(fontsize=7, framealpha=0.85, loc="upper center")
    if "carrier_hz" in res:
        up, lo = states
        a, b = axes[1, 1], axes[1, 2]
        a.plot(deg, res["carrier_hz"] / 1e3, color="#222222", lw=1.2, marker="o", ms=2.0)
        se._finish(a, xlabel, f"E_0({state_label(up)}) − E_0({state_label(lo)}) (kHz)",
                   f"Carrier part from the {pair} pair")
        b.plot(deg, res["zpe_diff_hz"], color="#222222", lw=1.2, marker="o", ms=2.0)
        se._finish(b, xlabel, "Difference of E_0 − V_min (Hz)", "Zero-point part of the carrier")
    else:
        axes[1, 1].set_visible(False)
        axes[1, 2].set_visible(False)
    for ax in axes.ravel():
        if not ax.get_visible():
            continue
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_xticks(np.arange(-180, 181, 30), minor=True)
        if current_deg is not None:
            ax.axvline(current_deg, color="#999999", lw=0.8, ls="--")
    lines = [f"Light-shift energies of the {pair}-pair wells versus the polarization angle ({settings_text})",
             "energies from the free atom at the bias field; differences: "
             f"{state_label(states[0])} minus {state_label(states[-1])}; dashed: current angle"]
    if current_deg is not None:
        i = int(np.argmin(np.abs(deg - current_deg)))
        lines.append(f"at {deg[i]:+.1f} deg: " + ";   ".join(
            f"{state_label(s)} V_min {res['states'][s]['v_min_hz'][i] / 1e3:.3f} kHz, "
            f"E_0 − V_min {res['states'][s]['zpe_hz'][i] / 1e3:.3f} kHz" for s in states))
    fig.suptitle("\n".join(lines), fontsize=8)


def draw_wells(fig, se, res, reference):
    """Wells tab: potentials, vibrational levels and ground-state densities of the
    edited pair, from each state's own minimum or on one absolute axis."""
    fig.clear()
    a, b = fig.subplots(2, 1)
    pair = res["settings"]["edited"]
    data = res["sp"]["axes"][pair]
    levels = data["levels"]
    upper, lower = list(levels)[:2]
    k_shown = shown_lower_site(data, lower)
    wells = [(upper, levels[upper]["psi"], levels[upper]["x0_m"]),
             (lower, levels[lower]["sites"][k_shown]["psi"], levels[lower]["sites"][k_shown]["x0_m"])]
    x_nm = levels[upper]["x"] * 1e9
    dx_nm = x_nm[1] - x_nm[0]
    centres = [c * 1e9 for _, _, c in wells]
    lo_nm, hi_nm = min(centres) - WINDOW_MARGIN_NM, max(centres) + WINDOW_MARGIN_NM
    sel = (x_nm >= lo_nm) & (x_nm <= hi_nm)
    common = reference == REFERENCES[1]
    spacing_nm = res["spacing_m"][pair] * 1e9
    y_lo, y_hi = np.inf, -np.inf
    for (state, psi, centre), c_nm in zip(wells, centres):
        d = levels[state]
        col = se.STATE_COLOURS.get(state, "#444444")
        zero = 0.0 if common else d.get("v_min_hz", d["v_hz"].min())
        v = d["v_hz"] - zero
        a.plot(x_nm[sel], v[sel] / 1e3, color=col, lw=1.1, label=state_label(state))
        own = sel & (np.abs(x_nm - c_nm) < spacing_nm / 2)          # this well only
        energies = d["energies_hz"][:LEVELS_SHOWN] - zero
        for e in energies:
            inside = np.flatnonzero(own & (v <= e))
            if inside.size:
                a.hlines(e / 1e3, x_nm[inside[0]], x_nm[inside[-1]], color=col, lw=0.7, alpha=0.75)
        v_min = d.get("v_min_hz", d["v_hz"].min()) - zero
        y_lo = min(y_lo, v_min)
        y_hi = max(y_hi, energies[-1] + 0.15 * (energies[-1] - v_min))
        b.plot(x_nm[sel], psi[0][sel] ** 2 / dx_nm, color=col, lw=1.1,
               label=state_label(state) + (", n = 0" if state == upper else f", n = 0, site {k_shown:+d}"))
        b.axvline(centre * 1e9, color=col, lw=0.8, ls="--")
    zero_shown = common and 0.0 < y_lo < y_hi - y_lo     # the free-atom zero is near: include it
    if zero_shown:
        y_lo = 0.0
        a.axhline(0.0, color="#444444", lw=0.8, ls=":")
    span = y_hi - y_lo
    # headroom above the data for the note
    a.set_ylim((y_lo - 0.04 * span) / 1e3 if common else 0.0, (y_hi + 0.45 * span) / 1e3)
    b.set_ylim(0.0, 1.8 * max(float(np.max(psi[0] ** 2)) for _, psi, _ in wells) / dx_nm)
    for ax in (a, b):
        ax.set_xlim(lo_nm, hi_nm)
    xlabel = f"Position along the {pair}-pair fringe direction (nm)"
    ylabel = "Energy from the free atom (kHz)" if common else "Potential above its minimum (kHz)"
    se._finish(a, xlabel, ylabel, f"Wells of the {pair} pair: potential and vibrational levels")
    se._finish(b, xlabel, "Probability density (1/nm)", f"Vibrational ground states of the {pair} pair")
    a.legend(loc="upper right", fontsize=7, framealpha=0.85)
    b.legend(loc="upper right", fontsize=7, framealpha=0.85)
    theta = res["settings"]["pairs"][pair]["theta_deg"]
    se._note(a, f"θ = {theta:+.1f} deg; f = {data['f_upper_hz'] / 1e3:.2f} kHz ({state_label(upper)}), "
                f"{data['f_lower_hz'] / 1e3:.2f} kHz ({state_label(lower)})\n"
                + ("energies from the free atom at the bias field (each state's own Zeeman energy)"
                   + ("; dotted: zero" if zero_shown else "") if common
                   else "each state drawn from its own minimum"))
    note = []
    for state in (upper, lower):
        d = levels[state]
        v_min = d.get("v_min_hz", d["v_hz"].min())
        note.append(f"{state_label(state)}: V_min {v_min / 1e3:.3f} kHz, E_0 {d['energies_hz'][0] / 1e3:.3f} kHz, "
                    f"E_0 − V_min {(d['energies_hz'][0] - v_min) / 1e3:.3f} kHz")
    pur = [data.get("purity_upper"), data.get("purity_lower")]
    if all(p is not None for p in pur):
        note.append(f"bias-field |F, m⟩ fraction at the minimum: {pur[0]:.3f} / {pur[1]:.3f}")
    sep = (wells[1][2] - wells[0][2]) * 1e9
    note.append(f"x({state_label(lower)}, site {k_shown:+d}) − x({state_label(upper)}) = {sep:+.1f} nm"
                + ("" if k_shown == 0 else f"; {state_label(lower)} well of its own site at "
                   f"{data['displacement_m'] * 1e9:+.0f} nm from {state_label(upper)}"))
    note.append("dashed: centre of each density")
    se._note(b, "\n".join(note))


# ------------------------------------------------------------------ GUI
class AngleView:
    def __init__(self, app):
        self.app, self.se = app, app.se
        tk, ttk = app.tk, app.ttk
        self.cache = {}                  # (energies_key, step) -> result
        self.run_id = 0
        self.pending = None              # (run_id, key) being computed
        self.handle = None
        self.t_start = 0.0

        # -- Energies vs angle tab
        self.panel = app.add_tab("energies", "Energies vs angle", self._draw,
                                 save_name=lambda: f"energies_vs_angle_{self.app.settings['edited']}")
        c = self.panel.controls
        ttk.Label(c, text="Angle step:").pack(side="left", padx=(4, 2))
        self.step_var = tk.StringVar(value=f"{DEFAULT_STEP_DEG:g} deg")
        box = ttk.Combobox(c, textvariable=self.step_var, state="readonly", width=8,
                           values=[f"{s:g} deg" for s in ANGLE_STEPS_DEG])
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Button(c, text="Stop", command=self.stop).pack(side="left", padx=6)
        self.status = ttk.Label(c, text="", foreground="#666666")
        self.status.pack(side="left", padx=6)
        ttk.Label(c, text="edited pair; recomputed when a setting other than the angle changes",
                  foreground="#666666").pack(side="left", padx=6)

        # -- Wells tab: energy reference
        w = app.p_wells.controls
        ttk.Label(w, text="Energy reference:").pack(side="left", padx=(4, 2))
        self.ref_var = tk.StringVar(value=REFERENCES[0])
        box = ttk.Combobox(w, textvariable=self.ref_var, state="readonly", width=40, values=REFERENCES)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._redraw_wells())
        app.p_wells.draw(self._draw_wells)

        # -- table rows (per pair: current (nominal))
        app.add_table_row("site of the nearest |3,−3⟩ well (spacings)",
                          lambda sp, ax, r: float(sp["axes"][ax]["nearest_site"]), "{:+.0f}")
        app.add_table_row("distance to the nearest |3,−3⟩ well (nm)",
                          lambda sp, ax, r: sp["axes"][ax]["nearest_displacement_m"] * 1e9, "{:+.1f}")
        app.add_table_row("line strength in the carrier and Δn = ±1",
                          lambda sp, ax, r: sp["axes"][ax]["strength_in_lines"], "ratio")
        app.add_table_row("centroid of the whole line strength (kHz)",
                          lambda sp, ax, r: sp["axes"][ax]["centroid_offset_hz"] / 1e3, "{:+.2f}")

        app.listeners.append(self.on_result)
        app.book.nb.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed(), add="+")

    # .. state
    def step(self):
        try:
            return float(self.step_var.get().split()[0])
        except ValueError:
            return DEFAULT_STEP_DEG

    def key(self):
        return energies_key(self.app.settings), self.step()

    def current_deg(self):
        s = self.app.settings
        return float(s["pairs"][s["edited"]]["theta_deg"])

    def shown(self):
        return self.app.book.current_key() == "energies"

    # .. computing
    def refresh(self):
        """Draw from the cache, or start the computation for the current settings."""
        key = self.key()
        if key in self.cache:
            self.panel.mark_dirty()
            self.panel.draw_if_visible()
            return
        if self.pending is not None and self.pending[1] == key:
            return
        self.start(key)

    def start(self, key):
        self.stop(quiet=True)
        self.run_id += 1
        rid = self.run_id
        settings = self.se.copy_settings(self.app.settings)
        thetas = angle_grid(key[1])
        chunks = [thetas[i:i + CHUNK_ANGLES] for i in range(0, thetas.size, CHUNK_ANGLES)]
        self.pending = (rid, key)
        self.t_start = time.perf_counter()
        self.status.configure(text=f"0 / {thetas.size} angles")
        self.panel.mark_dirty()
        self.panel.draw_if_visible()
        self._queue(rid, key, settings, thetas, chunks, [])

    def _queue(self, rid, key, settings, thetas, chunks, done_chunks):
        i = len(done_chunks)
        n_before = sum(c.size for c in chunks[:i])

        def work(progress, cancelled):
            return energies_for(self.se, settings, chunks[i], cancelled=lambda: cancelled() or rid != self.run_id)

        def finished(value):
            if rid != self.run_id or value is None:
                return
            done_chunks.append(value)
            n_done = n_before + chunks[i].size
            self.status.configure(text=f"{n_done} / {thetas.size} angles")
            if len(done_chunks) < len(chunks):
                self._queue(rid, key, settings, thetas, chunks, done_chunks)
                return
            if len(self.cache) >= CACHE_SIZE:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = merge_chunks(done_chunks, thetas)
            self.pending = None
            self.status.configure(text=f"{thetas.size} angles in {time.perf_counter() - self.t_start:.1f} s")
            self.panel.mark_dirty()
            self.panel.draw_if_visible()
        self.handle = self.app.run_background(work, finished)

    def stop(self, quiet=False):
        if self.handle is not None:
            self.handle.cancel()
        self.run_id += 1
        if self.pending is not None and not quiet:
            self.status.configure(text="stopped")
        self.pending = None

    # .. events
    def on_result(self, result):
        if self.shown():
            self.refresh()               # draws the new dashed line, or recomputes
        else:
            self.panel.mark_dirty()

    def _tab_changed(self):
        if self.shown():
            self.refresh()

    # .. drawing
    def _settings_text(self):
        s = self.app.settings
        ax = s["edited"]
        p, g = s["pairs"][ax], s["glob"]
        return (f"{ax} pair: depth {p['depth']:.2f} x full, ER {p['er']:.1e}, h = {int(p['handedness']):+d}, "
                f"arm offsets {p['static_um']:.1f} / {p['rotating_um']:.1f} µm; "
                f"B = {g['b_gauss']:.4f} G at {g['b_angle_deg']:.1f} deg")

    def _draw(self, fig):
        res = self.cache.get(self.key())
        try:
            waiting = ("Computing the energies over −180 … 180 deg (progress above)." if self.pending
                       else "Stopped; change a setting or the angle step to compute again.")
            draw_energies(fig, self.se, res, self.current_deg(), self._settings_text(), waiting)
        except Exception:
            traceback.print_exc()
            raise

    def _draw_wells(self, fig):
        draw_wells(fig, self.se, self.app.result, self.ref_var.get())

    def _redraw_wells(self):
        self.app.p_wells.mark_dirty()
        self.app.p_wells.draw_if_visible()


def attach(app):
    return AngleView(app)
