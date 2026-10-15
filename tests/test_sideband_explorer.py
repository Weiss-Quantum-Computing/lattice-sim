"""The sideband explorer's model functions (no window is opened)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from lattice_sim.constants import H, KB
from lattice_sim.spectra.sidebands import nominal_site, spectrum

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sideband_explorer.py"
_spec = importlib.util.spec_from_file_location("sideband_explorer", _PATH)
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)


def test_default_settings_are_the_nominal_site():
    site = se.site_from(se.default_settings())
    nominal = nominal_site()
    for a, b in zip(site.axes, nominal.axes):
        assert a.axis == b.axis
        assert a.trap_frequency_hz == pytest.approx(b.trap_frequency_hz, rel=1e-12)
        assert a.spacing_m == pytest.approx(b.spacing_m, rel=1e-12)
        assert a.theta == pytest.approx(b.theta, rel=1e-12)
        assert (a.extinction_ratio, a.handedness) == (b.extinction_ratio, b.handedness)
        assert (a.static_amplitude, a.rotating_amplitude) == (1.0, 1.0)
    assert np.allclose(site.b_lab_gauss, nominal.b_lab_gauss)
    assert site.temperature_k == nominal.temperature_k


def test_level_count_keeps_the_dropped_thermal_population_small():
    for t_uk in (0.5, 1.0, 2.0, 5.0):
        n = se.n_levels_for(t_uk * 1e-6, [12.5e3])
        r = math.exp(-H * 12.5e3 / (KB * t_uk * 1e-6))
        assert r ** n <= se.DROPPED_POPULATION or n == se.MAX_LEVELS
    assert se.n_levels_for(0.0, [12.5e3]) == 6


def test_explorer_spectrum_matches_the_library():
    s = se.default_settings()
    res = se.compute(s)
    site = se.site_from(s)
    ref = spectrum(site, max_dn=1)
    assert res["sp"]["carrier_shift_hz"] == pytest.approx(ref["carrier_shift_hz"], abs=1e-6)
    plus, minus = se.sideband_ratios(res["sp"], "x")
    assert 0.1 < plus < 0.3 and 0.03 < minus < 0.15      # 5.4 deg, 1 uK


def test_scan_over_the_polarization_angle():
    sc = se.scan(se.default_settings(), "theta_deg", [0.0, 5.4], cache={})
    assert sc["r_x"][0] < 1e-8 < 1e-3 < sc["r_x"][1]   # no sidebands at theta = 0
    assert sc["r_y"][0] == pytest.approx(sc["r_y"][1])   # the other pairs are untouched
    assert sc["current"] == 5.4


def test_gradients_vanish_along_the_fringes_for_a_polarization_offset():
    s = se.default_settings()
    s["pairs"]["x"]["rotating_um"] = 15.0
    assert se.needs_gradients(s)
    g = se.compute_gradients(s, cache={})
    assert g["carrier_hz_per_m_fringe"] == pytest.approx(0.0, abs=1e-6)
    assert abs(g["carrier_hz_per_m_pol"]) > 1.0


@pytest.mark.parametrize("direction, dalpha", [("pol", 0.0), ("fringe", 0.0), ("fringe", 0.3)])
def test_gradient_geometry_centre_is_the_explorer_site(direction, dalpha):
    from lattice_sim.spectra.beams import spectra_along
    s = se.default_settings()
    s["offset_dir"] = direction
    s["pairs"]["x"].update(static_um=10.0, rotating_um=-5.0, dalpha_deg=dalpha)
    direct = se.compute(s)["sp"]
    site, axis, geom = se.gradient_geometry(s)
    centre = spectra_along(site, axis, geom, [np.zeros(3)])[0]
    assert centre["axes"]["x"]["f_lower_hz"] == pytest.approx(direct["axes"]["x"]["f_lower_hz"], rel=1e-9)
    assert centre["carrier_shift_hz"] == pytest.approx(direct["carrier_shift_hz"], abs=1e-6)
