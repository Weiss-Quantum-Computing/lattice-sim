"""Sideband-explorer extension: the microwave pulse of a real frequency scan.

The explorer's line-strength spectrum is what an infinitely weak, infinitely
long pulse would show. A lab scan is a series of DDS pulses (AFP, Blackman
or square, with the firmware's envelope, sweep and 1/2/8 us time grid) whose
set frequency steps across the lines; the signal is the population moved to
the lower hyperfine state. This module adds

* the settings group "Microwave pulse" (shape, length, power in W as in the
  pulse file, AFP half sweep range, the lab's Blackman pi-power calibration,
  a shot-to-shot detuning spread, and the set frequency of a parked pulse);
* the Spectrum-tab quantity "transfer probability (pulse)";
* the "Pulse" tab: Rabi frequency and drive frequency against time, the
  response of one line (carrier and Delta n = +1 couplings of the ground
  state) against detuning, and a power scan at the set frequency (the lab's
  pi-power calibration);
* table rows: the transfer with the pulse parked on each pair's Delta n = +1
  and Delta n = -1 line (the temperature check);
* the lattice-grid quantity "transfer at the set microwave frequency".

The physics and the firmware conventions (with file:line references) are in
``lattice_sim.spectra.microwave``. The response table of a pulse is built on
the worker thread (``app.run_background``) whenever the pulse changes; the
spectrum, rows and grid quantity read it. Functions above ``attach`` need no
window and are tested in ``tests/test_microwave.py``.
"""

import time

import numpy as np

from lattice_sim.spectra import microwave as mw
from lattice_sim.spectra.sidebands import collapse_lines

MODE = "transfer probability (pulse)"
GRID_QUANTITY = "transfer at the set microwave frequency"
SHAPE_LABELS = {"afp": "AFP (pulse type 1)", "blackman": "Blackman (pulse type 2)", "square": "square (pulse type 3)"}
DETUNING_MAX_HZ = 80e3          # table reach: lines up to 2 trap frequencies beyond the plotted +/-1.25 f
MAX_STATES = 1000               # initial 3D states kept per spectrum (most populated first)
POWER_SCAN_POINTS = 41
POWER_SCAN_MAX_FACTOR = 4.0     # power scan from 0 to 4 x the set power (see power_scan_top)
REBUILD_DELAY_MS = 150
FIELD_SLOPE_HZ_PER_MG = 2.452e3  # |d f / d B| of |4,-4> -> |3,-3> at the magic field (findings 2026-10-08)


def lab_defaults():
    """Per-shape (length ms, power W, HSR kHz) the lab uses (``mw.LAB_PULSES``)."""
    out = {}
    for shape, (length, power, hsr) in mw.LAB_PULSES.items():
        if power is None:
            power = mw.pi_power(shape, length)
        out[shape] = dict(duration_ms=length * 1e3, power_w=power, hsr_khz=hsr * 1e-3)
    return out


def default_state():
    d = lab_defaults()
    return dict(shape="afp", memory=d, cal_w=mw.CAL_BLACKMAN_PI_POWER_W, spread_hz=0.0, set_khz=None,
                **d["afp"])


def pulse_from(state):
    """mw.Pulse of the extension state (power -> Rabi through the calibration)."""
    rabi = float(mw.rabi_from_power(state["power_w"], state["cal_w"]))
    return mw.Pulse(state["shape"], state["duration_ms"] * 1e-3, rabi, state["hsr_khz"] * 1e3)


def ground_lines(lines):
    """Lines of the most populated initial state: {(axis, dn): (detuning_hz, s)}; carrier key ('', 0)."""
    g = lines.state == lines.state[np.argmax(lines.population)]
    return {(str(a), int(d)): (float(f), float(s))
            for a, d, f, s in zip(lines.axis[g], lines.dn[g], lines.detuning_hz[g], lines.scale[g])}


def line_frequency(sp, axis, dn):
    """Detuning (Hz) from the free-atom line of a pair's sideband (population-weighted mean)."""
    if axis == "":
        return float(sp["carrier_shift_hz"])
    off, wt = collapse_lines(sp["lines"], axis).get((axis, dn), (np.nan, 0.0))
    return float(sp["carrier_shift_hz"] + off)


def visible_reach_khz(response, scales, rel=2e-3, pad=1.25):
    """Half-width (kHz) of the detuning range where any of the single-line curves exceeds rel x its peak."""
    reach = 0.0
    for s in scales:
        y = response.row(s)
        big = np.nonzero(y > rel * max(float(y.max()), 1e-12))[0]
        if big.size:
            reach = max(reach, float(np.max(np.abs(response.detuning_hz[big]))))
    reach = pad * reach if reach > 0 else float(response.detuning_hz[-1])
    return min(reach, float(response.detuning_hz[-1])) * 1e-3


def power_scan_top(state, ground, f_set_hz):
    """Highest power (W) of the power scan: 4 x the set power and, for square and
    Blackman pulses, at least 2.5 x the pi power of the ground-state line nearest
    the set frequency; at most POWMAX."""
    top = POWER_SCAN_MAX_FACTOR * state["power_w"]
    if state["shape"] != "afp" and ground:
        f, s = min(ground.values(), key=lambda v: abs(v[0] - f_set_hz))
        if s > 0:
            top = max(top, 2.5 * mw.pi_power(state["shape"], state["duration_ms"] * 1e-3, state["cal_w"], scale=s))
    return min(top, mw.POWMAX_W)


def calibration_note():
    """Which pi-power calibration assumption the absolute powers and Rabi
    frequencies use (``mw.CAL_IS_BARE_ATOM``; an open question, see
    ``lattice_sim.spectra.microwave``)."""
    if mw.CAL_IS_BARE_ATOM:
        return ("calibration assigned to the free atom (s = 1; open: if it was taken on the lattice carrier, "
                "s = 0.84, every Rabi frequency is 1.19x lower)")
    return "calibration assigned to the lattice carrier"


def first_maximum(x, y):
    """(x, y, interior) of the first local maximum of a sampled curve; without
    one, the largest point and interior = False."""
    y = np.asarray(y)
    for i in range(1, y.size - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            return float(x[i]), float(y[i]), True
    i = int(np.argmax(y))
    return float(x[i]), float(y[i]), False


def draw_pulse_tab(fig, se, pulse, response, ground, power_scan, f_set_hz, power_w, notes=""):
    """The Pulse tab: envelope, sweep, single-line response, power scan (2 x 2)."""
    fig.clear()
    (a, b), (c, d) = fig.subplots(2, 2)
    dt, amp, f0, f1 = mw.pulse_segments(pulse)
    t = np.concatenate([[0.0], np.cumsum(dt)]) * 1e3
    rabi = pulse.rabi_hz * amp * 1e-3
    a.stairs(rabi, t, color=se.CURRENT_COLOUR, lw=1.2, baseline=None)
    a.set_xlim(0, t[-1])
    a.set_ylim(0, 1.45 * max(float(rabi.max()), 1e-6))
    se._finish(a, "Time from the pulse start (ms)", "Rabi frequency Ω/2π (kHz)",
               "Rabi frequency of the set pulse, coupling s = 1")
    se._note(a, f"{SHAPE_LABELS[pulse.shape]}, {pulse.duration_s * 1e3:.3f} ms, {power_w:.4g} W\n"
                f"peak {pulse.rabi_hz * 1e-3:.3f} kHz, area {mw.pulse_area(pulse):.3f} cycles\n"
                + ("calibration: free atom (s = 1)" if mw.CAL_IS_BARE_ATOM else "calibration: lattice carrier"))
    tt = np.repeat(t, 2)[1:-1]
    ff = np.column_stack([f0, f1]).ravel() * 1e-3
    b.plot(tt, ff, color=se.CURRENT_COLOUR, lw=1.2)
    b.set_xlim(0, t[-1])
    span = max(abs(pulse.hsr_hz) * 1e-3, 0.5) if pulse.shape == "afp" else 1.0
    b.set_ylim(-1.5 * span, 1.5 * span)
    se._finish(b, "Time from the pulse start (ms)", "Drive frequency − set frequency (kHz)",
               "Drive frequency of the set pulse")
    corner = "top-left" if pulse.hsr_hz > 0 or pulse.shape != "afp" else "top-right"
    if pulse.shape == "afp":
        mid = float(np.interp(0.5 * t[-1], tt, ff)) * 1e3
        se._note(b, f"HSR {pulse.hsr_hz * 1e-3:+.2f} kHz, 8 µs tics\n"
                    f"offset at mid-pulse {mid:+.1f} Hz", corner)
    else:
        se._note(b, "constant frequency", corner)
    if response is None:
        c.text(0.5, 0.5, "building the pulse response ...", transform=c.transAxes, ha="center", va="center",
               color="#666666")
        c.set_xlim(-1, 1)
        c.set_ylim(0, 1.3)
    else:
        x = response.detuning_hz * 1e-3
        keys = [("", 0)]
        side = [(ax, 1) for ax in "xyz" if (ax, 1) in ground]
        if side:
            keys.append(max(side, key=lambda k: ground[k][1]))
        texts = []
        for key in keys:
            s = ground[key][1]
            y = response.row(s)
            name = "carrier" if key[0] == "" else f"{key[0]} Δn = +1"
            colour = se.CURRENT_COLOUR if key[0] == "" else se.PAIR_COLOURS[key[0]]
            c.plot(x, y, color=colour, lw=1.2, label=f"{name}, s = {s:.3f}")
            peak, fwhm = mw.response_width(response.detuning_hz, y)
            texts.append(f"{name}: peak {peak:.3f}, FWHM {fwhm * 1e-3:.2f} kHz")
        c.set_xlim(-visible_reach_khz(response, [ground[k][1] for k in keys]),
                   visible_reach_khz(response, [ground[k][1] for k in keys]))
        c.set_ylim(0, 1.3)
        c.legend(loc="upper right", fontsize=7, framealpha=0.85)
        if response.spread_hz > 0:
            texts.append(f"detuning spread {response.spread_hz:.0f} Hz rms")
        se._note(c, "\n".join(texts))
    se._finish(c, "Set frequency − line frequency (kHz)", "Transferred population",
               "Response of one line of the ground state")
    if power_scan is None:
        d.text(0.5, 0.5, "computing the power scan ...", transform=d.transAxes, ha="center", va="center",
               color="#666666")
        d.set_xlim(0, 1)
        d.set_ylim(0, 1.3)
        text = []
    else:
        pw, y = power_scan
        d.plot(pw, y, color="#222222", marker="o", ms=2.5, lw=1.0)
        d.axvline(power_w, color="#999999", lw=0.8, ls="--")
        d.set_xlim(0, pw[-1])
        d.set_ylim(0, 1.3)
        px, py, interior = first_maximum(pw, y)
        text = [f"set frequency {f_set_hz * 1e-3:+.3f} kHz from the free-atom line",
                f"{'first maximum' if interior else 'largest value'} {py:.3f} at {px:.4g} W; dashed: set power"]
    if notes:
        text.append(notes)
    if text:
        se._note(d, "\n".join(text))
    se._finish(d, "Microwave power (W)", "Transferred population, all lines",
               "Power scan at the set frequency")


# ===================================================================== window
class MicrowaveExtension:
    def __init__(self, app):
        self.app = app
        self.se = se = app.se
        self.state = default_state()
        self.response = None            # Response for the current pulse and spread
        self._raw = None                # the same before the detuning spread
        self._job = None
        self._scan_job = None
        self._after = None
        self.power_scan = None
        self._lines = {}                # id(sp) -> (sp, lines)
        self._curves = {}               # (id(sp), id(response), f0, f1, n) -> y
        self.status_text = ""
        if self.state["set_khz"] is None:
            self.state["set_khz"] = app.ref["sp"]["carrier_shift_hz"] * 1e-3

        tk, ttk = app.tk, app.ttk
        Param = se.Param

        class Plain(Param):
            def fmt(self, v):
                return f"{v:.4g}"

        grp = app.add_settings_group("Microwave pulse")
        app.help_label(grp, "The pulse of a lab frequency scan: the Spectrum tab's \"show: " + MODE + "\" "
                       "draws the population it moves to |3,−3⟩ versus its set frequency. Values as in the DDS "
                       "pulse file.")
        row = ttk.Frame(grp)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Pulse shape").pack(side="left")
        self.shape_var = tk.StringVar(value=SHAPE_LABELS[self.state["shape"]])
        box = ttk.Combobox(row, textvariable=self.shape_var, values=list(SHAPE_LABELS.values()),
                           state="readonly", width=24)
        box.pack(side="right")
        box.bind("<<ComboboxSelected>>", lambda e: self._shape_chosen())
        app.help_label(grp, "AFP: sin² amplitude with a √(1 − sin⁴) frequency sweep, near-complete transfer "
                       "over the swept range. Blackman: Blackman-window amplitude at a fixed frequency, the "
                       "lab's π pulses. Square: constant amplitude.")
        self.params = dict(
            duration_ms=Plain("duration_ms", "Pulse length", "ms", 0.02, 10.0, self.state["duration_ms"], 3,
                              "The pulse file's Length. The DDS floors it to whole steps: Blackman amplitude every "
                              "2 µs, AFP tics of 8 µs, square in 1 µs.", log=True),
            power_w=Plain("power_w", "Power", "W", 1e-4, mw.POWMAX_W, self.state["power_w"], 4,
                          "The pulse file's Power (peak power). The field, hence the Rabi frequency, goes as "
                          "√power. 5.188 W is the top of the DDS power table.", log=True),
            hsr_khz=Param("hsr_khz", "AFP half sweep range (HSR)", "kHz", -20.0, 20.0, self.state["hsr_khz"], 2,
                          "The AFP sweeps from f − HSR to f + HSR; the lab uses −4 kHz (a downward sweep). "
                          "Not used by Blackman and square pulses."),
            cal_w=Plain("cal_w", "Calibration: π power of a 0.12 ms Blackman", "W", 0.01, mw.POWMAX_W,
                        self.state["cal_w"], 4,
                        "Power scan on |4,−4⟩ → |3,−3⟩ (0.306 W since April 2022). It sets the Rabi frequency "
                        "of every pulse; it is assigned to the free atom (coupling s = 1).", log=True),
            spread_hz=Param("spread_hz", "Detuning spread (rms)", "Hz", 0.0, 2000.0, 0.0, 0,
                            "Shot-to-shot detuning noise, averaged as a Gaussian: 245 Hz is 0.1 mG of field noise "
                            "(−2.45 kHz/mG). 0 = none."),
            set_khz=Param("set_khz", "Set frequency of a parked pulse", "kHz", -40.0, 40.0, self.state["set_khz"], 3,
                          "From the free-atom line. Used by the power scan and by the lattice-grid view; "
                          "\"Park on\" puts it on a line of the current spectrum."),
        )
        self.rows = {}
        for key in ("duration_ms", "power_w", "hsr_khz", "cal_w", "spread_hz", "set_khz"):
            self.rows[key] = app.make_param_row(grp, self.params[key], self._changed)
        brow = ttk.Frame(grp)
        brow.pack(fill="x", pady=(6, 0))
        self.pi_btn = ttk.Button(brow, text="π power (s = 1)", command=lambda: self._pi_power(False))
        self.pi_btn.pack(side="left")
        self.pi_car_btn = ttk.Button(brow, text="π power on the carrier", command=lambda: self._pi_power(True))
        self.pi_car_btn.pack(side="left", padx=6)
        prow = ttk.Frame(grp)
        prow.pack(fill="x", pady=(6, 0))
        ttk.Label(prow, text="Park on").pack(side="left")
        names = ["carrier"] + [f"{ax} Δn = {dn:+d}" for ax in "xyz" for dn in (1, -1)]
        self.park_var = tk.StringVar(value="carrier")
        ttk.Combobox(prow, textvariable=self.park_var, values=names, state="readonly", width=14).pack(
            side="left", padx=4)
        ttk.Button(prow, text="Set", command=self._park).pack(side="left")
        self.readout = app.help_label(grp, "")
        self._update_buttons()

        app.add_spectrum_mode(MODE, self.mode)
        self.panel = app.add_tab("pulse", "Pulse", self.draw, figsize=(8, 4.6), save_name="microwave_pulse")
        app.add_table_row("transfer, pulse parked on Δn = +1 (pulse tab settings)",
                          lambda sp, ax, res: self.parked(sp, ax, 1), "{:.3f}")
        app.add_table_row("transfer, pulse parked on Δn = −1 (hot atoms)",
                          lambda sp, ax, res: self.parked(sp, ax, -1), "{:.3f}")
        app.listeners.append(self._new_result)
        app.grid_quantities[GRID_QUANTITY] = self.grid_value
        self._schedule(0)

    # .. state
    def pulse(self):
        return pulse_from(self.state)

    def _changed(self, key):
        self.state[key] = float(self.rows[key].get())
        if key == "spread_hz":
            if self._raw is not None:
                self.response = self._raw.with_spread(self.state["spread_hz"])
                self._refresh(table_changed=True)
            return
        if key == "set_khz":
            self.power_scan = None
            self._refresh(table_changed=False)
            return
        self._schedule()

    def _shape_chosen(self):
        shape = {v: k for k, v in SHAPE_LABELS.items()}[self.shape_var.get()]
        if shape == self.state["shape"]:
            return
        mem = self.state["memory"]
        mem[self.state["shape"]] = {k: self.state[k] for k in ("duration_ms", "power_w", "hsr_khz")}
        self.state["shape"] = shape
        for k, v in mem[shape].items():
            self.state[k] = v
            self.rows[k].set(v)
        self._update_buttons()
        self._schedule()

    def _update_buttons(self):
        st = ["disabled"] if self.state["shape"] == "afp" else ["!disabled"]
        self.pi_btn.state(st)
        self.pi_car_btn.state(st)

    def _pi_power(self, on_carrier):
        if self.state["shape"] == "afp":
            return
        s = 1.0
        if on_carrier:
            lines = self.lines(self.app.result["sp"])
            s = ground_lines(lines).get(("", 0), (0.0, 0.0))[1]
            if not s > 0:
                # wells apart (mid angles): no carrier on any site, so no pi power
                self.readout.configure(text="no carrier at these settings (coupling s = 0): no π power")
                return
        p = mw.pi_power(self.state["shape"], self.state["duration_ms"] * 1e-3, self.state["cal_w"], scale=s)
        p = min(p, mw.POWMAX_W)
        self.rows["power_w"].set(p)
        self._changed("power_w")

    def _park(self):
        name = self.park_var.get()
        sp = self.app.result["sp"]
        if name == "carrier":
            f = line_frequency(sp, "", 0)
        else:
            ax, dn = name[0], int(name.split("=")[1].replace("−", "-"))
            f = line_frequency(sp, ax, dn)
        if np.isfinite(f):
            self.rows["set_khz"].set(f * 1e-3)
            self._changed("set_khz")

    # .. the response table (worker thread)
    def _schedule(self, delay=REBUILD_DELAY_MS):
        if self._after is not None:
            self.app.root.after_cancel(self._after)
        self._after = self.app.root.after(delay, self._rebuild)

    def _rebuild(self):
        self._after = None
        if self._job is not None:
            self._job.cancel()
        if self._scan_job is not None:
            self._scan_job.cancel()
        try:
            pulse = self.pulse()
        except ValueError as err:
            self.readout.configure(text=f"pulse not valid: {err}")
            return
        spread = self.state["spread_hz"]
        self.response = self._raw = None
        self.power_scan = None
        self.readout.configure(text="building the pulse response ...")

        def build(progress, cancelled):
            return mw.build_response(pulse, DETUNING_MAX_HZ, 0.0, progress=progress, cancelled=cancelled)

        def done(resp, pulse=pulse, spread=spread):
            if resp is None or resp.pulse != self.pulse():
                return
            self._raw = resp
            self.response = resp.with_spread(self.state["spread_hz"])
            self._refresh(table_changed=True)

        def progress(i, n):
            self.readout.configure(text=f"building the pulse response ... {100 * i // max(n, 1)} %")
        self._job = self.app.run_background(build, done, progress)

    def _refresh(self, table_changed):
        self._curves.clear()
        self._show_readout()
        self.panel.mark_dirty()
        self.panel.draw_if_visible()
        if table_changed:
            self.app.redraw_spectrum()
            if hasattr(self.app, "quantity_changed"):
                self.app.quantity_changed(GRID_QUANTITY)
            self.app.request_update()       # table rows, then the power scan
        else:
            self._maybe_power_scan()

    def _show_readout(self):
        r = self.response
        p = self.pulse()
        lines = self.lines(self.app.result["sp"])
        g = ground_lines(lines)
        text = (f"Peak Rabi frequency (s = 1): {p.rabi_hz * 1e-3:.3f} kHz; area {mw.pulse_area(p):.3f} cycles "
                f"({calibration_note()}). "
                f"Ground-state carrier coupling s = {g.get(('', 0), (0.0, 0.0))[1]:.3f}.")
        if r is not None:
            text += (f" Response table {r.q.shape[0]} × {r.q.shape[1]}, ±{r.detuning_hz[-1] * 1e-3:.1f} kHz, "
                     f"{r.seconds:.2f} s.")
        text += (f" {lines.n_states} initial states kept ({100 * lines.kept_population:.2f} % of the population);"
                 f" {100 * mw.strength_outside_lines(lines):.1f} % of their line strength is in lines not drawn "
                 "(sidebands on two pairs at once, |Δn| > 2, levels above the cut).")
        self.readout.configure(text=text)

    # .. quantities
    def lines(self, sp):
        hit = self._lines.get(id(sp))
        if hit is not None and hit[0] is sp:
            return hit[1]
        lines = mw.lines_from_spectrum(sp, max_states=MAX_STATES)
        if len(self._lines) > 8:
            self._lines.clear()
        self._lines[id(sp)] = (sp, lines)
        return lines

    def mode(self, result, f_hz):
        r = self.response
        if r is None:
            return None, ""
        sp = result["sp"]
        key = (id(sp), id(r), float(f_hz[0]), float(f_hz[-1]), len(f_hz))
        y = self._curves.get(key)
        if y is None:
            y = r.scan(self.lines(sp), f_hz)
            if len(self._curves) > 8:
                self._curves.clear()
            self._curves[key] = y
        p = self.pulse()
        lines = self.lines(sp)
        note = (f"{p.shape} {p.duration_s * 1e3:.3f} ms, peak Rabi {p.rabi_hz * 1e-3:.2f} kHz"
                + (f", HSR {p.hsr_hz * 1e-3:+.2f} kHz" if p.shape.lower().startswith("afp") else "")
                + f"; {lines.n_states} initial states ({100 * lines.kept_population:.2f} %), "
                f"{100 * mw.strength_outside_lines(lines):.1f} % of the strength in lines not drawn"
                + ("; calibration: free atom (s = 1)" if mw.CAL_IS_BARE_ATOM else ""))
        return y, "Transferred population after the pulse", note

    def parked(self, sp, axis, dn):
        r = self.response
        if r is None:
            return float("nan")
        f = line_frequency(sp, axis, dn)
        return r.transfer_at(self.lines(sp), f) if np.isfinite(f) else float("nan")

    def grid_value(self, sp):
        # runs on the worker thread; self.response is replaced by the Tk thread in done(). Attribute
        # assignment is atomic in CPython, so at worst one grid uses the previous pulse, and
        # quantity_changed(GRID_QUANTITY) then invalidates that grid.
        r = self.response
        if r is None:
            return float("nan")
        return r.transfer_at(mw.lines_from_spectrum(sp, max_states=MAX_STATES), self.state["set_khz"] * 1e3)

    def _new_result(self, result):
        self.power_scan = None
        self._show_readout()
        self.panel.mark_dirty()
        self.panel.draw_if_visible()
        self._maybe_power_scan()

    # .. power scan (worker thread)
    def _maybe_power_scan(self):
        if self.power_scan is not None or not self.panel.is_visible():
            return
        if self._scan_job is not None:
            self._scan_job.cancel()
        pulse = self.pulse()
        lines = self.lines(self.app.result["sp"])
        f_set = self.state["set_khz"] * 1e3
        spread = self.state["spread_hz"]
        p0 = self.state["power_w"]
        top = power_scan_top(self.state, ground_lines(lines), f_set)
        powers = np.linspace(0.0, top, POWER_SCAN_POINTS)
        key = (pulse, f_set, spread, id(lines))
        self._scan_key = key

        def run(progress, cancelled):
            t0 = time.perf_counter()
            y = mw.power_scan(lines, pulse, f_set, powers / p0, spread_hz=spread, cancelled=cancelled)
            return None if y is None else (powers, y, time.perf_counter() - t0)

        def done(value, key=key):
            if value is None or key != self._scan_key:
                return
            self.power_scan = value[:2]
            self.panel.mark_dirty()
            self.panel.draw_if_visible()
        self._scan_job = self.app.run_background(run, done)

    # .. Pulse tab
    def draw(self, fig):
        try:
            pulse = self.pulse()
        except ValueError as err:
            self.app.labplot.placeholder(fig, f"pulse not valid: {err}")
            return
        lines = self.lines(self.app.result["sp"])
        self._maybe_power_scan()
        notes = ""
        if lines.harmonic:
            notes = "harmonic ladders (no level energies given)"
        draw_pulse_tab(fig, self.se, pulse, self.response, ground_lines(lines), self.power_scan,
                       self.state["set_khz"] * 1e3, self.state["power_w"], notes)


def attach(app):
    return MicrowaveExtension(app)
