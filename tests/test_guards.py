"""Inputs outside what a function was built for fail or warn instead of passing silently."""

import warnings

import numpy as np
import pytest

from lattice_sim.fields.extinction import ExtinctionProfile
from lattice_sim.motion.fast_gaussian import HarmonicScenario, propagate_scenarios
from lattice_sim.motion.transport1d import PotentialTable, load_waveform


def _row(leg, direction, segment, rot, er, method="dip"):
    return dict(method=method, leg=float(leg), direction=direction, segment=segment, rotation_deg=float(rot),
                er=float(er), lower_bound=0.0)


ROWS = [
    _row(1, "static", "rest", 0.04, 3000), _row(1, "away", "up 1", -45, 4000), _row(1, "away", "up 1", -90, 3e4),
    _row(1, "static", "hold 1", -180, 1500), _row(1, "back", "down 1", -90, 2e4),
    _row(1, "static", "after 1", 0.3, 2000),
    _row(2, "away", "up 2", -14.8, 2572), _row(2, "away", "up 2", -90, 3e4), _row(2, "static", "hold 2", -180, 1900),
]


def test_later_legs_are_anchored_by_the_rest_before_them():
    p = ExtinctionProfile.from_rows(ROWS, 2, "away")
    assert np.rad2deg(p.theta[0]) == pytest.approx(0.3) and p.er[0] == 2000     # 'after 1', not 2572 held
    p1 = ExtinctionProfile.from_rows(ROWS, 1, "away")
    assert np.rad2deg(p1.theta[0]) == pytest.approx(0.04) and p1.er[0] == 3000
    back = ExtinctionProfile.from_rows(ROWS, 1, "back")
    assert np.rad2deg(back.theta).tolist() == pytest.approx([0.3, 90, 180])


def test_blank_rows_are_dropped_and_short_profiles_refused():
    rows = ROWS + [_row(1, "away", "up 1", -60, np.nan), dict(_row(1, "away", "up 1", -75, 5000), leg=np.nan)]
    p = ExtinctionProfile.from_rows(rows, 1, "away")
    assert np.all(np.isfinite(p.er)) and np.rad2deg(p.theta).tolist() == pytest.approx([0.04, 45, 90, 180])
    with pytest.raises(ValueError, match="at least 2"):
        ExtinctionProfile.from_rows([_row(3, "away", "up 3", -90, 3e4)], 3, "away", include_static=False)


def test_profile_warns_outside_the_measured_angles():
    p = ExtinctionProfile(np.deg2rad([14.8, 180.0]), np.array([2572.0, 1900.0]), np.zeros(2, bool), "leg 2 away")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p(np.deg2rad([20.0, 179.5]))                                     # inside: no warning
    with pytest.warns(UserWarning, match="leg 2 away"):
        assert p(0.0) == pytest.approx(2572.0)                           # held, but said


def test_minimum_on_a_hump_returns_nan_frequency():
    """A narrow hump between the coarse samples makes the fine quadratic fit concave: no
    harmonic frequency (it used to come back as 0 Hz)."""
    from lattice_sim.atom.ground_state import magic_field
    from lattice_sim.fields.lattice_pair import LatticePair
    from lattice_sim.fields.potential import StatePotential

    class Hump(StatePotential):
        def potential(self, state, x, theta):
            x = np.asarray(x, float)
            return 1e4 * (x / 1e-6) ** 2 + 1e3 * np.exp(-((x - 1.8e-9) / 2e-9) ** 2)

    model = Hump(LatticePair(), magic_field() * np.array([1.0, 0.0, 0.0]))
    x0, f = model.minimum((3, -3), 0.0)
    assert np.isnan(f) and abs(x0) < 1e-8
    plain = StatePotential(LatticePair(), magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0]))
    assert plain.minimum((3, -3), 0.0)[1] == pytest.approx(10e3, rel=1e-3)     # a real trap is unaffected


def test_waveform_with_a_return_leg_is_refused(tmp_path):
    path = tmp_path / "ramp.csv"
    t = np.arange(0.0, 101.0, 1.0)
    up = np.clip(t / 50.0, 0, 1) * 5400.0
    ripple = up + 0.5 * np.sin(t)                                       # sub-volt noise: clamped
    path.write_text("time_us,voltage_V\n" + "".join(f"{a},{b}\n" for a, b in zip(t, ripple)), encoding="utf-8")
    _, theta, v = load_waveform(path)
    assert np.all(np.diff(v) >= 0) and theta[-1] == pytest.approx(np.pi * v[-1] / 5400.0)
    back = np.where(t <= 50, up, 5400.0 * (1 - (t - 50) / 50.0))         # forward + return
    path.write_text("time_us,voltage_V\n" + "".join(f"{a},{b}\n" for a, b in zip(t, back)), encoding="utf-8")
    with pytest.raises(ValueError, match="running maximum"):
        load_waveform(path)
    assert load_waveform(path, max_drop_v=np.inf)[2][-1] == pytest.approx(5400.0)


def test_harmonic_scenario_keeps_the_sign_and_mirrors_give_the_same_population():
    th = np.linspace(0, np.pi, 31)
    x0 = -2.4e-6 * np.sin(th / 2) ** 2                                   # a state moving to -x
    sc = HarmonicScenario(th, x0, np.full(th.size, 10e3))
    assert sc.x0(np.pi) == pytest.approx(-2.4e-6)
    t = np.arange(0.0, 300.0, 0.5)
    tt = np.pi * np.sin(0.5 * np.pi * t / t[-1]) ** 2
    om = sc.omega(tt)[None, :]
    p = propagate_scenarios(t, sc.x0(tt)[None, :], om)
    p_mirror = propagate_scenarios(t, -sc.x0(tt)[None, :], om)
    assert np.allclose(p, p_mirror, atol=1e-12) and p[0, -1] < 0.999    # excited, identically


def test_potential_table_warns_outside_its_angles():
    table = PotentialTable(np.array([0.0, np.pi]), np.linspace(-1e-6, 1e-6, 5), np.ones((2, 5)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table.at(np.pi)                                                  # the end itself is fine
    with pytest.warns(UserWarning, match="outside"):
        table.at(np.pi + 0.01)


def test_storage_light_shift_scales_as_the_square_root_of_the_site_intensity():
    from lattice_sim.spin.echo import Sequence, coefficients

    class Pair:
        kappa_b_hz_per_mg = -2000.0
        storage_ls_hz = 30.0

        def de0(self, th):
            return 0 * th

        def s(self, th):
            return 0 * th

        def ddtheta(self, th, eta):
            return 0 * th

    g = Sequence([("wait", 100.0)], dt_us=10.0).build()
    for a in (0.5, 0.9, 1.0):
        c = coefficients(g, Pair(), lambda th: 0 * th, a=a)
        assert c["static"] == pytest.approx(30.0 * np.sqrt(a))           # LS ~ f ~ sqrt(I)
        # d(static)/d(ln a), the intensity-noise coefficient
        assert c["intensity"] == pytest.approx(0.5 * 30.0 * np.sqrt(a))


def test_with_alignment_calls_can_be_chained():
    from lattice_sim.spectra.sidebands import amplitude_from_offset, nominal_site, with_alignment
    site = with_alignment(nominal_site(), "x", static_offset_m=20e-6)
    site = with_alignment(site, "x", crossing_half_angle_deg=60.0)        # keeps the offset
    x = [c for c in site.axes if c.axis == "x"][0]
    assert x.static_amplitude == pytest.approx(amplitude_from_offset(20e-6)) and x.rotating_amplitude == 1.0
    x = [c for c in with_alignment(site, "x", static_offset_m=0.0).axes if c.axis == "x"][0]
    assert x.static_amplitude == 1.0
