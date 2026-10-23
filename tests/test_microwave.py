"""Microwave pulses of the lab's DDS (``lattice_sim.spectra.microwave``)."""

import numpy as np
import pytest

from lattice_sim.constants import H, MASS_CS
from lattice_sim.spectra import microwave as mw
from lattice_sim.spectra.sidebands import collapse_lines, nominal_site, spectrum, thermal_weights, vibrational_levels

PI_06 = mw.pi_power("blackman", 0.6e-3)
BLACKMAN_06 = mw.Pulse("blackman", 0.6e-3, float(mw.rabi_from_power(PI_06)))
BLACKMAN_012 = mw.Pulse("blackman", 0.12e-3, float(mw.rabi_from_power(0.306)))
AFP_LAB = mw.Pulse("afp", 3e-3, float(mw.rabi_from_power(0.08)), -4e3)
SQUARE_06 = mw.Pulse("square", 0.6e-3, float(mw.rabi_from_power(mw.pi_power("square", 0.6e-3))))


@pytest.fixture(scope="module")
def nominal_lines():
    sp = spectrum(nominal_site(), max_dn=1)
    return sp, mw.lines_from_spectrum(sp)


def harmonic_pair(f_u=19e3, f_l=19.02e3, x0=35e-9, n=6, temperature_k=1e-6):
    """Two displaced 1D harmonic wells: energies (from each ground state), signed overlaps, populations."""
    x = np.linspace(-1.2e-6, 1.2e-6, 2401)
    vu = 0.5 * MASS_CS * (2 * np.pi * f_u) ** 2 * x ** 2 / H
    vl = 0.5 * MASS_CS * (2 * np.pi * f_l) ** 2 * (x - x0) ** 2 / H
    eu, pu = vibrational_levels(x, vu, n, half_window_m=1.1e-6)
    el, pl = vibrational_levels(x, vl, n, half_window_m=1.1e-6)
    return eu - eu[0], el - el[0], pl @ pu.T, thermal_weights(f_u, temperature_k, n)


def lines_1d(eu, el, o, w):
    n_l, n_u = o.shape
    return mw.make_lines((eu[None, :] - el[:, None]).ravel(), np.abs(o).ravel(),
                         np.repeat(w[None, :], n_l, 0).ravel(), np.repeat(np.arange(n_u)[None, :], n_l, 0).ravel())


# ------------------------------------------------------------- firmware
def test_firmware_time_grid_and_envelopes():
    dt, amp, f0, f1 = mw.pulse_segments(mw.Pulse("blackman", 0.12e-3, 1.0))
    assert dt.size == 60 and np.allclose(dt, 2e-6)                  # 1 + Time2Tics(4 L) tics, 2 words each
    assert amp[-1] == pytest.approx(0.0, abs=1e-12) and amp[29] == pytest.approx(1.0, abs=1e-12)
    assert np.sum(amp * dt) == pytest.approx(0.42 * 0.12e-3, rel=1e-12)   # amplitude area 0.42 T
    assert np.all(f0 == 0) and np.all(f1 == 0)
    dt, amp, f0, f1 = mw.pulse_segments(mw.Pulse("square", 0.6e-3, 1.0))
    assert dt.tolist() == [pytest.approx(600e-6)]
    dt, amp, f0, f1 = mw.pulse_segments(AFP_LAB)
    n = 375                                                           # floor(3 ms / 8 us)
    assert dt.size == n - 1 and np.allclose(dt, 8e-6)
    i = np.arange(1, n)
    assert np.allclose(amp, np.sin(np.pi * (i - 1) / (n - 1)) ** 2)   # sqrt of the sin^4 power
    kappa = 8e-6 * (n - 1) / 3e-3
    hsr = AFP_LAB.hsr_hz
    assert f0[0] == pytest.approx(-hsr)                               # starts at f - HSR
    assert f1[-1] == pytest.approx(-hsr + 2 * kappa * hsr)            # ends near f + HSR
    mid = np.interp(0.5, np.arange(n) / (n - 1), np.r_[f0, f1[-1]])
    assert mid == pytest.approx(-(1 - kappa) * hsr, abs=1e-6)          # +10.7 Hz for HSR = -4 kHz
    assert -(1 - kappa) * hsr == pytest.approx(10.67, abs=0.01)


def test_afp_law_keeps_the_effective_field_constant():
    tau = np.linspace(0, 1, 201)
    detuning = mw._afp_frequency(tau, 1.0)
    assert np.allclose(np.sin(np.pi * tau) ** 4 + detuning ** 2, 1.0)   # (Omega/Omega_pk)^2 + (Delta/HSR)^2 = 1


def test_power_calibration_and_pi_powers():
    assert mw.rabi_from_power(0.306) == pytest.approx(1 / (2 * 0.42 * 0.12e-3), rel=1e-12)     # 9.92 kHz
    assert mw.pi_power("blackman", 0.12e-3) == pytest.approx(0.306, rel=1e-12)
    assert PI_06 == pytest.approx(0.306 / 25, rel=1e-12)            # 5x longer -> 1/25 power
    assert mw.transfer(BLACKMAN_06, 1.0, 0.0) == pytest.approx(1.0, abs=1e-9)
    assert mw.transfer(BLACKMAN_012, 1.0, 0.0) == pytest.approx(1.0, abs=1e-9)
    half = BLACKMAN_06.with_rabi(mw.rabi_from_power(PI_06 / 4))     # P/4 -> pi/2
    assert mw.transfer(half, 1.0, 0.0) == pytest.approx(0.5, abs=1e-9)
    assert mw.transfer(SQUARE_06, 1.0, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_hop_weights():
    for m in range(-4, 5):
        assert sum(mw.hop_weight(m, m + q) for q in (-1, 0, 1) if abs(m + q) <= 3) == 28   # rank-1 sum rule
    assert mw.hop_weight(-4, -3) == 28 and mw.hop_weight(0, 0) == 16 and mw.hop_weight(-2, -3) == 1
    with pytest.raises(ValueError):
        mw.hop_weight(-4, -2)


# ------------------------------------------------------------ two-level
def test_square_pulse_against_rabi_formula_and_ode():
    d = np.linspace(-6e3, 6e3, 25)
    for s in (0.2, 0.7, 1.0):
        exact = mw.transfer_square_analytic(SQUARE_06, s, d)
        assert np.max(np.abs(mw.transfer(SQUARE_06, s, d) - exact)) < 1e-12
    for s, x in ((0.7, 0.0), (1.0, 1300.0), (0.3, -2500.0)):
        assert mw.transfer_reference(SQUARE_06, s, x) == pytest.approx(
            float(mw.transfer_square_analytic(SQUARE_06, s, x)), abs=1e-8)


@pytest.mark.parametrize("pulse, points, tol", [
    (BLACKMAN_06, ((1.0, 0.0), (0.4, 1500.0), (0.9, -2700.0)), 1e-8),
    (BLACKMAN_012.with_rabi(2.3 * BLACKMAN_012.rabi_hz), ((1.0, 0.0), (0.6, 7000.0)), 1e-8),
    (AFP_LAB, ((1.0, 0.0), (0.4, 1500.0), (0.9, -3500.0), (0.3, 4100.0)), 2e-5),
])
def test_propagator_against_solve_ivp(pulse, points, tol):
    for s, d in points:
        assert mw.transfer(pulse, s, d) == pytest.approx(mw.transfer_reference(pulse, s, d), abs=tol)


def test_zero_coupling_and_conservation():
    s = np.linspace(0, 1, 7)[:, None]
    d = np.linspace(-8e3, 8e3, 9)[None, :]
    for pulse in (BLACKMAN_06, AFP_LAB, SQUARE_06):
        assert np.all(mw.transfer(pulse, 0.0, d) == 0)
        up, low = mw._propagate(pulse, s, d, both=True)
        assert np.max(np.abs(up + low - 1)) < 1e-12


def test_afp_transfers_completely_and_saturates_in_power():
    assert mw.transfer(AFP_LAB, 1.0, 0.0) > 0.9999
    assert np.min(mw.transfer(AFP_LAB, 1.0, np.linspace(-2e3, 2e3, 9))) > 0.9995
    assert np.min(mw.transfer(AFP_LAB, 1.0, np.linspace(-2.5e3, 2.5e3, 11))) > 0.99     # 0.9933 at the ends
    # above threshold the transfer does not depend on power (unlike a pi pulse)
    p = [mw.transfer(AFP_LAB.with_rabi(AFP_LAB.rabi_hz * np.sqrt(k)), 1.0, 0.0) for k in (1, 2, 4, 8)]
    assert max(p) - min(p) < 1e-3
    pi = [mw.transfer(BLACKMAN_06.with_rabi(BLACKMAN_06.rabi_hz * np.sqrt(k)), 1.0, 0.0) for k in (1, 2, 4)]
    assert min(pi) < 0.6
    assert mw.transfer(AFP_LAB, 1.0, 12e3) < 1e-5                    # outside the sweep


# ------------------------------------------------------------- tables
SQUARE_3MS_2PI = mw.Pulse("square", 3e-3, float(mw.rabi_from_power(4 * mw.pi_power("square", 3e-3))))


@pytest.mark.parametrize("pulse", [AFP_LAB, BLACKMAN_06, BLACKMAN_06.with_rabi(3 * BLACKMAN_06.rabi_hz), SQUARE_3MS_2PI])
def test_table_interpolation_matches_direct_propagation(pulse):
    r = mw.build_response(pulse, 60e3)
    rng = np.random.default_rng(3)
    s = rng.uniform(0, 1, 1500)
    d = rng.uniform(r.detuning_hz[0], r.detuning_hz[-1], 1500)
    err = np.abs(r.at(s, d) - mw.transfer(pulse, s, d))
    assert err.max() < 0.01 and np.sqrt(np.mean(err ** 2)) < 1e-3
    if pulse.shape != "square":
        assert r.probability()[:, [0, -1]].max() < 1e-6              # nothing beyond the table


def test_detuning_spread_two_ways():
    r = mw.build_response(BLACKMAN_06, 60e3, spread_hz=400.0)
    d = np.linspace(-3e3, 3e3, 13)
    for s in (0.3, 1.0):
        direct = mw.transfer(BLACKMAN_06, s, d, spread_hz=400.0)        # Gauss-Hermite
        assert np.max(np.abs(r.at(s, d) - direct)) < 5e-3
    assert r.at(1.0, 0.0) < 0.95                                         # the spread lowers the peak


def test_scan_matches_line_by_line_sum(nominal_lines):
    sp, lines = nominal_lines
    f = np.linspace(-25e3, 25e3, 801)
    for pulse in (BLACKMAN_06, AFP_LAB):
        r = mw.build_response(pulse, 60e3)
        y = r.scan(lines, f)
        direct = np.array([r.transfer_at(lines, x) for x in f[::40]])
        assert np.max(np.abs(y[::40] - direct)) < 3e-3
        assert y.max() <= lines.kept_population + 1e-9
    r = mw.build_response(BLACKMAN_06, 60e3)
    assert np.max(np.abs(r.scan(lines, f) - r.scan_sum(lines, f))) < 0.02     # resolved lines: plain sum


def test_lines_reproduce_the_spectrum_weights(nominal_lines):
    sp, lines = nominal_lines
    assert lines.kept_population > 0.998            # 13 levels per pair at 1 uK: states below 1e-5 dropped
    lines = mw.lines_from_spectrum(sp, min_population=1e-9)
    assert lines.kept_population > 0.9999
    for (ax, dn), (off, wt) in collapse_lines(sp["lines"]).items():
        sel = (lines.axis == ax) & (lines.dn == dn)
        assert np.sum(lines.population[sel] * lines.scale[sel] ** 2) == pytest.approx(wt, rel=2e-4)
    car = lines.axis == ""
    assert np.sum(lines.population[car] * lines.scale[car] ** 2) == pytest.approx(sp["carrier_weight"], rel=2e-4)
    ground = (lines.state == lines.state[np.argmax(lines.population)])
    gc = ground & car
    assert lines.detuning_hz[gc][0] == pytest.approx(sp["carrier_shift_hz"], abs=1e-6)
    expect = np.prod([np.sqrt(sp["axes"][a]["franck_condon"][0, 0]) for a in "xyz"])
    assert lines.scale[gc][0] == pytest.approx(expect, rel=1e-12)
    gx = ground & (lines.axis == "x") & (lines.dn == 1)
    lv = sp["axes"]["x"]["levels"]
    e_u, e_l = lv[(4, -4)]["energies_hz"], lv[(3, -3)]["energies_hz"]
    assert lines.detuning_hz[gx][0] == pytest.approx(sp["carrier_shift_hz"] + (e_l[0] - e_l[1]), abs=1e-6)


@pytest.fixture(scope="module", params=[0.97, 1.0], ids=["0.97pi", "pi"])
def large_angle_lines(request):
    """The x pair turned to 0.97 pi / pi, where its carrier lies on the neighbouring site k = +1."""
    from dataclasses import replace
    site = nominal_site()
    axes = [replace(c, theta=request.param * np.pi) if c.axis == "x" else c for c in site.axes]
    sp = spectrum(replace(site, axes=axes), max_dn=1)
    return sp, mw.lines_from_spectrum(sp, max_dn=1, min_population=1e-9)


def test_lines_reproduce_the_spectrum_weights_at_large_angle(large_angle_lines):
    sp, lines = large_angle_lines
    x = sp["axes"]["x"]
    assert x["diagonal_sites"][0] < 1e-6 and x["diagonal_sites"][1] > 0.7       # carrier on site k = +1
    w = lines.population * lines.scale ** 2
    car = lines.axis == ""
    assert np.sum(w[car]) == pytest.approx(sp["carrier_weight"], rel=2e-4)
    for (ax, dn), (off, wt) in collapse_lines(sp["lines"]).items():
        sel = (lines.axis == ax) & (lines.dn == dn)
        assert np.sum(w[sel]) == pytest.approx(wt, rel=2e-4, abs=1e-9)
    # the lines hold the spectrum's whole listed strength (carrier + sidebands, all sites)
    assert np.sum(w) == pytest.approx(sp["strength_in_lines"], rel=2e-4)
    assert mw.strength_outside_lines(lines) == pytest.approx(1 - np.sum(w) / lines.kept_population, abs=1e-12)


def test_scan_sees_the_neighbour_site_carrier(large_angle_lines):
    """A weak pulse's scan integrates to sum p s^2 (the strength of every line, carrier
    included), and a pi pulse on the carrier transfers: the scan does not read 0 near pi."""
    sp, lines = large_angle_lines
    weak = BLACKMAN_06.with_rabi(0.05 * BLACKMAN_06.rabi_hz)          # linear regime: P = s^2 x shape
    r = mw.build_response(weak, 60e3)
    f = np.linspace(-60e3, 60e3, 24001)
    unit = np.sum(r.row(1.0)) * r.step_hz                              # integral of one s = 1 line
    total = np.sum(r.scan_sum(lines, f)) * (f[1] - f[0]) / unit
    assert total == pytest.approx(np.sum(lines.population * lines.scale ** 2), rel=2e-3)
    assert total == pytest.approx(sp["strength_in_lines"], rel=2e-3)
    full = mw.build_response(BLACKMAN_06, 60e3)
    assert full.transfer_at(lines, sp["carrier_shift_hz"]) > 0.3
    ext = _extension()
    s_car = ext.ground_lines(lines)[("", 0)][1]
    assert s_car > 0.5                                                  # pi power on the carrier is finite
    assert np.isfinite(mw.pi_power("blackman", 0.6e-3, scale=s_car))
    assert ("s = 1" in ext.calibration_note()) == mw.CAL_IS_BARE_ATOM


def test_power_scan_agrees_with_the_table(nominal_lines):
    sp, lines = nominal_lines
    f_set = sp["carrier_shift_hz"]
    ps = mw.power_scan(lines, BLACKMAN_06, f_set, [0.0, 0.25, 1.0])
    r = mw.build_response(BLACKMAN_06, 60e3)
    assert ps[0] == 0.0
    assert ps[2] == pytest.approx(r.transfer_at(lines, f_set), abs=3e-3)
    one = mw.make_lines([0.0], [1.0], [1.0])
    assert mw.power_scan(one, BLACKMAN_06, 0.0, [0.25, 1.0]) == pytest.approx([0.5, 1.0], abs=1e-9)


def test_lines_of_one_state_never_transfer_more_than_its_population():
    two = mw.make_lines([0.0, 500.0], [1.0, 1.0], [1.0, 1.0], state=[0, 0])
    p = mw.combine_lines(two, np.array([[0.8], [0.5]]))
    assert p[0] == pytest.approx(1 - 0.2 * 0.5)
    sep = mw.make_lines([0.0, 500.0], [1.0, 1.0], [0.5, 0.5])
    assert mw.combine_lines(sep, np.array([[0.8], [0.5]]))[0] == pytest.approx(0.65)


# ---------------------------------------------- independent-line check
def coupled_vs_independent(pulse, f):
    eu, el, o, w = harmonic_pair()
    lines = lines_1d(eu, el, o, w)
    coupled = mw.coupled_transfer(eu, el, o, w, pulse, f)
    indep = np.array([mw.combine_lines(lines, mw.transfer(pulse, lines.scale, x - lines.detuning_hz)) for x in f])
    return coupled, indep


def test_independent_lines_agree_with_the_coupled_model_at_low_rabi():
    f = np.linspace(-24e3, 24e3, 49)
    c, i = coupled_vs_independent(BLACKMAN_06, f)          # Rabi 2 kHz, trap 19 kHz
    assert np.max(np.abs(c - i)) < 0.015
    c, i = coupled_vs_independent(SQUARE_06, f)
    assert np.max(np.abs(c - i)) < 0.01


def test_independent_lines_fail_at_full_power_and_for_the_lab_afp():
    f = np.linspace(-24e3, 24e3, 25)
    c, i = coupled_vs_independent(BLACKMAN_012, f)        # Rabi 9.9 kHz against 19 kHz
    assert np.max(np.abs(c - i)) > 0.04
    f = np.linspace(-24e3, -14e3, 11)                     # AFP over the dn = +1 sideband
    c, i = coupled_vs_independent(AFP_LAB, f)
    assert 0.01 < np.max(np.abs(c - i)) < 0.06            # the carrier's light shift on the sideband


def test_harmonic_ladders_for_spectra_without_levels(nominal_lines):
    import copy
    sp, lines = nominal_lines
    bare = copy.deepcopy(sp)
    for d in bare["axes"].values():
        d.pop("levels")
    approx = mw.lines_from_spectrum(bare)
    assert approx.harmonic and not lines.harmonic
    assert np.allclose(approx.scale, lines.scale) and np.allclose(approx.population, lines.population)
    w = lines.population * lines.scale ** 2
    assert np.sum(w * np.abs(approx.detuning_hz - lines.detuning_hz)) / w.sum() < 20.0     # 11 Hz
    r = mw.build_response(BLACKMAN_06, 60e3)
    f = np.linspace(-25e3, 25e3, 401)
    assert np.max(np.abs(r.scan(approx, f) - r.scan(lines, f))) < 0.01


def test_state_cap_keeps_the_most_populated_states(nominal_lines):
    sp, lines = nominal_lines
    capped = mw.lines_from_spectrum(sp, max_states=50)
    assert capped.n_states == 50 and capped.kept_population < lines.kept_population
    assert capped.state_population.min() >= np.sort(lines.state_population)[-50] - 1e-15


# ------------------------------------------------- explorer extension (no window)
def _extension():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "explorer_microwave.py"
    spec = importlib.util.spec_from_file_location("explorer_microwave", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extension_defaults_are_the_lab_pulses(nominal_lines):
    ext = _extension()
    state = ext.default_state()
    p = ext.pulse_from(state)
    assert (p.shape, p.duration_s, p.hsr_hz) == ("afp", 3e-3, -4e3)
    assert p.rabi_hz == pytest.approx(mw.rabi_from_power(0.08))
    bl = dict(state, shape="blackman", **state["memory"]["blackman"])
    assert mw.transfer(ext.pulse_from(bl), 1.0, 0.0) == pytest.approx(1.0, abs=1e-9)   # pi pulse at s = 1
    sp, lines = nominal_lines
    g = ext.ground_lines(lines)
    assert g[("", 0)][0] == pytest.approx(sp["carrier_shift_hz"], abs=1e-6)
    assert ext.first_maximum([0, 1, 2, 3], [0, 0.5, 1.0, 0.5])[:2] == (2.0, 1.0)
    assert ext.first_maximum([0, 1, 2], [0, 0.5, 0.9])[2] is False
    top = ext.power_scan_top(bl, g, g[("x", 1)][0])                 # covers the x sideband's pi power
    assert top >= 2.5 * mw.pi_power("blackman", 0.6e-3, scale=g[("x", 1)][1]) - 1e-12 or top == mw.POWMAX_W


def test_nominal_site_numbers():
    """Numbers quoted in docs/findings.md (nominal lattice, 1 uK, 13 levels as in the explorer)."""
    from dataclasses import replace
    sp = spectrum(replace(nominal_site(), n_levels=13), max_dn=1)
    lines = mw.lines_from_spectrum(sp, max_states=1000)
    g = lines.state == lines.state[np.argmax(lines.population)]
    s_car = float(lines.scale[g & (lines.axis == "")][0])
    assert s_car == pytest.approx(0.840, abs=0.002)
    assert mw.pi_power("blackman", 0.12e-3, scale=s_car) == pytest.approx(0.433, abs=0.002)
    ps = mw.power_scan(lines, BLACKMAN_06, sp["carrier_shift_hz"], [1.0, 2.25])
    assert ps == pytest.approx([0.66, 0.86], abs=0.01)
