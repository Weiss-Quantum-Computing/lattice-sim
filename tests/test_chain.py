"""Tests for the atom-chain collisional-gate model (lattice_sim.gates.chain)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

from lattice_sim.constants import A0
from lattice_sim.gates import chain as ch
from lattice_sim.gates.collision import (MU, CollisionModel, interaction_hz, oscillator_width,
                                         scattering_operator, uniform_lengths)

DATA = Path(__file__).resolve().parents[1] / "data"
P4_HOLDS_V = (4340.0, 4785.0, 5155.0)


@pytest.fixture(scope="module")
def calibration():
    return ch.calibration_model()


@pytest.fixture(scope="module")
def phase4_coarse():
    """The Phase 4 first-principles |3,+-3> model (measured ER, h = -1, 10 kHz) on a
    coarse angle grid that contains the Phase 4 hold angles as nodes (~3 s)."""
    holds = ch.voltage_to_theta(np.array(P4_HOLDS_V))
    thetas = np.unique(np.concatenate([np.linspace(0, 0.76 * np.pi, 8), holds,
                                       np.linspace(0.8 * np.pi, np.pi, 4)]))
    return ch.first_principles_model(3, -1, thetas=thetas, cache_dir=None)


def _large_gap_model():
    th = np.linspace(0, np.pi, 60)
    lat = np.full((th.size, 7), 1e9)                 # all other channels far detuned
    lat[:, list(MU).index(3)] = 0.0
    disp = 2.4187e-6 * np.sin(th / 2) ** 2
    return CollisionModel(3, th, disp, disp, np.full(th.size, 9e3), np.full(th.size, 8e3), lat)


# ------------------------------------------------------------ overlap
def test_calibration_model_reproduces_the_lab_overlap_table():
    # tools/overlap_vs_voltage.py: 1e-3 / 0.1 / 0.5 / 0.9 at 3800 / 4340 / 4785 / 5155 V,
    # 9.48 kHz at full overlap for 7 kHz transverse and 2800 a0, pi/2 in ~26 us
    cm = ch.calibration_model(transverse_hz=(7e3, 7e3))
    v = np.linspace(0, 5400, 54001)
    rel = cm.overlap(ch.voltage_to_theta(v)) / ch.full_overlap(cm)
    for level, volts, tol in ((1e-3, 3800, 5), (0.1, 4340, 2), (0.5, 4785, 2), (0.9, 5155, 2)):
        assert v[np.argmax(rel >= level)] == pytest.approx(volts, abs=tol)
    u_full = interaction_hz(2800 * A0, ch.full_overlap(cm))
    assert u_full / 1e3 == pytest.approx(9.48, abs=0.01)
    assert 0.25 / u_full * 1e6 == pytest.approx(26.4, abs=0.1)
    # end widths 87 nm, spacing 4.837 um
    _, _, s_plus, s_minus = ch.Tracks(cm)(np.pi)
    assert s_plus * 1e9 == pytest.approx(87, abs=1) and s_minus * 1e9 == pytest.approx(87, abs=1)
    assert cm.spacing_m * 1e6 == pytest.approx(4.837, abs=1e-3)


def test_chain_hold_overlap_matches_phase4(phase4_coarse):
    """U/h and separation at the Phase 4 hold voltages: the chain's sequence
    readout against the stored Phase 4 output (data/p4_results.json)."""
    with open(DATA / "p4_results.json", encoding="utf-8") as fh:
        p4 = json.load(fh)
    rows = {r["hold_V"]: r for r in p4["pairs"] if r["m"] == 3 and r["h"] == -1}
    for v in P4_HOLDS_V:
        s = ch.GateSettings(m=3, n_atoms=3, hold_v=v, ramp="smooth", smooth_ramp_us=300.0, lock_hold=False,
                            hold_us=10.0, n_frames=50, scan_points=9)
        res = ch.compute_sequence(phase4_coarse, s, hold_scan=False)
        assert res["hold"]["u_hz"] == pytest.approx(rows[v]["uniform 2800 a0"]["U_hold_hz"], rel=1e-4)
        assert res["hold"]["separation_m"] * 1e9 == pytest.approx(rows[v]["separation_nm"], abs=0.01)


def test_drawn_densities_overlap_equals_the_gaussian_formula(calibration):
    """Numerical overlap of the packet densities the window draws (atom j's
    |3,+m>, atom j+1's |3,-m>) against collision.overlap_density."""
    cm = calibration
    tr = ch.Tracks(cm)
    sites = ch.site_positions(4, cm.spacing_m)
    fy, fz = cm.transverse_hz                       # identical transverse Gaussians: O_y O_z
    transverse = 1 / (2 * np.pi * oscillator_width(fy) * oscillator_width(fz))
    for v in (3800.0, 4340.0, 4785.0, 5155.0, 5400.0):
        th = float(ch.voltage_to_theta(v))
        xp, xm, sp, sm = tr(th)
        x = np.linspace(sites[1] - 1.5e-6, sites[2] + 1.5e-6, 200001)
        n_left = ch.gaussian_density(x, sites[1] + xp, sp)
        n_right = ch.gaussian_density(x, sites[2] - xm, sm)
        numeric = trapezoid(n_left * n_right, x) * transverse
        assert numeric == pytest.approx(float(cm.overlap(th)), rel=1e-5)


def test_gaussian_overlap_agrees_with_numerical_eigenstates(phase4_coarse):
    """Second method: overlap of the anharmonic site ground states of the
    first-principles potentials (cf. scripts/p4_overlap_check.py)."""
    from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
    from lattice_sim.fields.lattice_pair import LatticePair
    from lattice_sim.fields.potential import StatePotential
    from lattice_sim.spectra.sidebands import vibrational_levels
    cm = phase4_coarse
    profile = ExtinctionProfile.from_rows(read_rampol_er(ch.MEASURED_ER_CSV), 1, "away")
    model = StatePotential(LatticePair(extinction_ratio=profile, handedness=-1), ch.phase4_b_vector(),
                           trap_frequency_hz=ch.PHASE4_TRAP_HZ)
    tr = ch.Tracks(cm)
    for v in (4340.0, 4785.0):
        th = float(ch.voltage_to_theta(v))
        xp, xm, sp, sm = tr(th)
        dens = []
        for state, centre in (((3, 3), xp), ((3, -3), -xm)):
            x = centre + np.linspace(-1.0e-6, 1.0e-6, 2001)
            _, psi = vibrational_levels(x, model.potential(state, x, th), n_levels=1, half_window_m=0.9e-6)
            dens.append((x, np.abs(psi[0]) ** 2 / (x[1] - x[0])))
        grid = np.linspace(xp - 1.0e-6, xp + 1.0e-6, 40001)          # left atom at 0, right atom at +a
        n_left = np.interp(grid, dens[0][0], dens[0][1], left=0, right=0)
        n_right = np.interp(grid - cm.spacing_m, dens[1][0], dens[1][1], left=0, right=0)
        numeric_x = trapezoid(n_left * n_right, grid)
        d = cm.spacing_m - xp - xm
        gauss_x = np.exp(-d**2 / (sp**2 + sm**2)) / np.sqrt(np.pi * (sp**2 + sm**2))
        assert gauss_x / numeric_x == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------------ phase
def test_leg_phase_equals_collision_evolve_when_gaps_are_large():
    cm = _large_gap_model()
    op = scattering_operator(uniform_lengths(2800.0))
    a_m = ch.effective_scattering_length_m(3, "uniform", 2800.0)
    assert a_m == pytest.approx(2800 * A0, rel=1e-12)
    rt, rth = ch.smooth_ramp(0.9 * np.pi, 200.0)
    for hold in (0.0, 37.3):
        t, th = ch.leg_trajectory(rt, rth, 0.9 * np.pi, hold)
        phi_evolve, pops = cm.evolve(t, th, op, dt_us=0.5)
        phi_chain = ch.leg_phase(cm, rt, rth, 0.9 * np.pi, hold, a_m)
        assert phi_chain == pytest.approx(phi_evolve, rel=2e-4)
        assert pops[cm.index] == pytest.approx(1.0, abs=1e-6)


def test_leg_phase_against_seven_state_model_with_real_gaps(phase4_coarse):
    """With the real lattice gaps, evolve() differs from int U dt only by the
    second-order shift from the exchanged pair |L:-3, R:+3>, ~U/gap."""
    s = ch.GateSettings(m=3, n_atoms=2, hold_v=4150.0, ramp="smooth", smooth_ramp_us=500.0, lock_hold=True,
                        n_frames=50, scan_points=9)
    res = ch.compute_sequence(phase4_coarse, s, hold_scan=False)
    chk = ch.seven_state_check(phase4_coarse, s, res)
    assert res["phi1"] == pytest.approx(np.pi / 2, rel=1e-3)
    assert chk["phi"] == pytest.approx(res["phi1"], rel=0.02)
    assert chk["leakage"] < 1e-3


def test_sequence_phases_and_hold_solver(calibration):
    for ramp in ("robust", "smooth"):
        s = ch.GateSettings(m=3, n_atoms=5, hold_v=4150.0, ramp=ramp, lock_hold=True, n_frames=200, scan_points=9)
        res = ch.compute_sequence(calibration, s, hold_scan=False)
        assert not res["hold_capped"] and res["hold_us"] > 0
        assert res["phi1"] == pytest.approx(np.pi / 2, rel=1e-3)
        assert res["phi2"] == pytest.approx(res["phi1"], rel=1e-6)
        # the leg phase from the sequence grid equals the single-leg integral
        assert res["phi1"] == pytest.approx(ch.leg_phase(calibration, res["ramp_t"], res["ramp_theta"],
                                                         res["theta_hold"], res["hold_us"], res["a_m"]), rel=1e-4)
        # timeline: pi/2 - leg - pi - leg - pi/2, theta = 0 outside the legs
        names = [p[0] for p in res["pulses"]]
        assert names == ["π/2", "π", "π/2"]
        assert res["t_end_us"] == pytest.approx(2 * res["leg_us"] + 2 * s.gap_us)
        assert res["leg_us"] == pytest.approx(2 * res["ramp_us"] + res["hold_us"])
        f = res["frames"]
        assert np.all(f["theta"][~f["in_leg"]] == 0.0)
        assert f["theta"].max() == pytest.approx(res["theta_hold"], rel=1e-9)
    # the hold-voltage scan (1 us step) at the set voltage gives the same leg phase
    s = ch.GateSettings(m=3, hold_v=4150.0, ramp="smooth", lock_hold=True, n_frames=50, scan_points=9)
    res = ch.compute_sequence(calibration, s, hold_scan=True)
    sc = res["hold_scan"]
    k = int(np.argmin(np.abs(sc["volts"] - 4150.0)))
    assert sc["volts"][k] == pytest.approx(4150.0)
    assert sc["phi_leg"][k] == pytest.approx(res["phi1"], rel=1e-4)
    # the ramp alone beyond pi/2: no hold
    s = ch.GateSettings(m=3, hold_v=4785.0, ramp="robust", lock_hold=True, n_frames=50, scan_points=9)
    res = ch.compute_sequence(calibration, s, hold_scan=False)
    assert res["hold_us"] == 0.0 and res["phi1"] > np.pi / 2


# ------------------------------------------------------------ Ramsey signal
def test_cos_rule_matches_state_vector_simulation():
    scan = np.linspace(0, 2 * np.pi, 25)
    rng = np.random.default_rng(3)
    # n = 0: an isolated atom has the bare fringe (1 - cos phase) / 2
    assert ch.chain_fringes(1, [], [], scan)[0] == pytest.approx((1 - np.cos(scan)) / 2, abs=1e-12)
    for n_atoms in (2, 3, 4, 7):
        for _ in range(3):
            p1, p2 = rng.uniform(0, 2 * np.pi, n_atoms - 1), rng.uniform(0, 2 * np.pi, n_atoms - 1)
            sim = ch.chain_fringes(n_atoms, p1, p2, scan)
            assert sim == pytest.approx(ch.cos_rule_fringes(n_atoms, p1, p2, scan), abs=1e-12)


def test_contrast_is_cos_to_the_number_of_neighbours():
    for n_atoms in (2, 3, 7):
        nb = ch.neighbour_counts(n_atoms)
        assert nb[0] == nb[-1] == 1 and np.all(nb[1:-1] == 2)
        for leg in (0.0, 0.3, np.pi / 4, np.pi / 2, 2.0):
            sim = ch.chain_fringes(n_atoms, leg, leg, np.array([0.0, np.pi]))
            contrast = sim[:, 1] - sim[:, 0]                    # P(pi) - P(0) = C
            assert contrast == pytest.approx(np.cos(leg) ** nb, abs=1e-12)   # cos^n(Phi/2), Phi = 2 leg
    # sqrt(ZZ) (pi/2 per leg): no fringe on any atom with a neighbour
    sim = ch.chain_fringes(5, np.pi / 2, np.pi / 2, np.linspace(0, 2 * np.pi, 13))
    assert np.ptp(sim, axis=1) == pytest.approx(np.zeros(5), abs=1e-12)


def test_window_module_loads_and_maps_explorer_settings():
    import importlib.util
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("explorer_chain", scripts / "explorer_chain.py")
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        spec = importlib.util.spec_from_file_location("sideband_explorer", scripts / "sideband_explorer.py")
        se = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(se)
    finally:
        sys.path.remove(str(scripts))
    d = ext.defaults_from_settings(se.default_settings(), se)
    site = se.site_from(se.default_settings())
    x = [c for c in site.axes if c.axis == "x"][0]
    assert d["trap_hz"] == pytest.approx(x.trap_frequency_hz)
    assert d["spacing_m"] == pytest.approx(x.spacing_m)
    assert d["er"] == pytest.approx(3e4) and d["handedness"] == -1
    assert np.linalg.norm(d["b_vector"]) == pytest.approx(1.39304, abs=1e-5)


def test_window_worker_keeps_polling_after_a_callback_error():
    """chain_explorer.Worker: a raising done/progress callback is reported and the poll re-arms
    (it used to stop, so the window showed 'computing' forever). No Tk: a fake widget."""
    import importlib.util
    import queue
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("chain_explorer", scripts / "chain_explorer.py")
        cx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cx)
    finally:
        sys.path.remove(str(scripts))

    class FakeWidget:
        def __init__(self):
            self.armed = []

        def after(self, ms, fn):
            self.armed.append(fn)

    widget, errors, seen = FakeWidget(), [], []
    worker = cx.Worker(widget, on_error=errors.append)
    try:
        def boom(value):
            raise RuntimeError("drawing failed")
        worker.results.put((boom, (1,)))
        worker.results.put((seen.append, (2,)))
        n_armed = len(widget.armed)
        worker._poll()
        assert seen == [2]                               # the result after the failing one still arrives
        assert errors == ["display failed - see the console"]
        assert len(widget.armed) == n_armed + 1          # re-armed
        # a job that raises on the worker thread is reported too
        done = queue.Queue()
        worker.run_background(lambda progress, cancelled: 1 / 0, done.put)
        for _ in range(200):
            if not worker.results.empty():
                break
            import time
            time.sleep(0.01)
        worker._poll()
        assert errors[-1] == "calculation failed - see the console" and done.empty()
    finally:
        worker.close()


def test_sequence_states_its_untested_assumptions(calibration):
    s = ch.GateSettings(m=3, hold_v=4150.0, ramp="robust", lock_hold=True, n_frames=20, scan_points=9)
    res = ch.compute_sequence(calibration, s, hold_scan=False)
    assert any("leg 1 only" in n for n in res["notes"]) and any("PLUS_MOVER_QUBIT" in n for n in res["notes"])
    # outside the model's angle grid the overlap holds its end value, as the packet tracks do
    th_end = calibration.thetas[-1]
    assert calibration.overlap(th_end + 0.2) == pytest.approx(float(calibration.overlap(th_end)), rel=1e-12)
    assert calibration.separation(-0.1) == pytest.approx(float(calibration.separation(calibration.thetas[0])))
