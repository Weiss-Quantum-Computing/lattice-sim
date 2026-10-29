"""Sideband-explorer extension: the "Molasses" group, which opens the
molasses window (``molasses_explorer.py``) on the same Tk root.

The window takes the explorer's current trap frequencies (x, y, z, from the
pair depths and crossing angles) as its full-depth lattice for live Monte
Carlo runs; the tables (data/molasses/*.jsonl) are always at the nominal
19 / 16 / 12.5 kHz. Its runs use the window's own worker thread, so a
minute-long Monte Carlo run does not block the spectrum.
"""


def defaults_from_settings(settings, se):
    site = se.site_from(settings)
    by_axis = {c.axis: c.trap_frequency_hz * 1e-3 for c in site.axes}
    return dict(trap_khz=tuple(float(by_axis[a]) for a in ("x", "y", "z")))


def attach(app):
    ttk = app.ttk
    grp = app.add_settings_group("Molasses (imaging / PGC)")
    app.help_label(grp, "Temperature, scattering rate, hopping rate and field sensitivity of the six-beam "
                        "molasses in the lattice against intensity and detuning (Monte Carlo tables and live "
                        "runs). Opens in its own window; the current trap frequencies set its lattice.")
    state = dict(win=None)

    def open_window():
        import molasses_explorer
        d = defaults_from_settings(app.settings, app.se)
        win = state["win"]
        if win is not None and win.alive():
            win.set_explorer_defaults(d)
            win.lift()
            return win
        state["win"] = molasses_explorer.MolassesWindow(app.root, defaults=d)
        return state["win"]

    button = ttk.Button(grp, text="Open the molasses window", command=open_window)
    button.pack(anchor="w", pady=(4, 0))
    state.update(open=open_window, button=button)
    return state
