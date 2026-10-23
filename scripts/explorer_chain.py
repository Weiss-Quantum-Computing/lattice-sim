"""Sideband-explorer extension: the "Atom chain" group, which opens the
atom-chain window (``chain_explorer.py``) on the same Tk root.

The window gets the explorer's current x-pair settings (trap frequency from
depth and crossing angle, spacing, extinction ratio, handedness, bias field)
as an extra lattice model, "First principles, explorer x pair", next to the
Phase 4 configuration (10 kHz x trap, measured extinction profile) and the
Transport-Optimizer calibration files. Its calculations run on the
window's own worker thread, not the explorer's (``run_background=None``),
so the ~30 s first build of a track cache does not block the spectrum; the
two workers share no cache objects (the chain window keeps its own models
and data/chain_tracks_*.npz), so running both at once is safe.
"""

import numpy as np


def defaults_from_settings(settings, se):
    """The chain window's defaults from the explorer's settings dict."""
    site = se.site_from(settings)
    x = [c for c in site.axes if c.axis == "x"][0]
    g = settings["glob"]
    phi = np.deg2rad(g["b_angle_deg"])
    return dict(trap_hz=float(x.trap_frequency_hz), spacing_m=float(x.spacing_m),
                er=float(x.extinction_ratio), handedness=int(x.handedness),
                b_vector=g["b_gauss"] * np.array([np.cos(phi), np.sin(phi), 0.0]))


def attach(app):
    ttk = app.ttk
    grp = app.add_settings_group("Atom chain (collisional gate)")
    app.help_label(grp, "A line of 2-7 atoms on neighbouring x sites through the spin-echoed collisional gate "
                        "(π/2, leg, π, leg, π/2): packet densities, neighbour overlap, collisional phase and the "
                        "Ramsey fringe of end and interior atoms. Opens in its own window; the current x-pair "
                        "settings become one of its lattice models.")
    state = dict(win=None)

    def open_window():
        import chain_explorer
        d = defaults_from_settings(app.settings, app.se)
        win = state["win"]
        if win is not None and win.alive():
            win.set_explorer_defaults(d)
            win.lift()
            return win
        # its own worker: a first track build takes ~30 s and must not hold up the spectrum
        state["win"] = chain_explorer.ChainWindow(app.root, defaults=d, run_background=None)
        return state["win"]

    button = ttk.Button(grp, text="Open the atom-chain window", command=open_window)
    button.pack(anchor="w", pady=(4, 0))
    state.update(open=open_window, button=button)
    return state
