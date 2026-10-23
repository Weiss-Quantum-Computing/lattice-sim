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
    """Over the slider ranges (depth 0.3-1.5, 0-5 uK, every pair) the level count keeps <= 1e-3 of
    the thermal population above the last level wherever it is below the cap; at the cap more is
    dropped, and that is what the status line reports (next test)."""
    depth = [p for p in se.PAIR_PARAMS if p.key == "depth"][0]
    temp = [p for p in se.GLOBAL_PARAMS if p.key == "temperature_uk"][0]
    capped = []
    for d in np.linspace(depth.lo, depth.hi, 7):
        for f_full in (19e3, 16e3, 12.5e3):
            f = f_full * math.sqrt(d)
            for t_uk in np.linspace(max(temp.lo, 0.25), temp.hi, 8):
                n = se.n_levels_for(t_uk * 1e-6, [f])
                r = math.exp(-H * f / (KB * t_uk * 1e-6))
                if n < se.MAX_LEVELS:
                    assert r ** n <= se.DROPPED_POPULATION
                else:
                    capped.append((d, f_full, t_uk, r ** n))
    assert se.n_levels_for(0.0, [12.5e3]) == 6
    # the cap is reached only below full depth and above 2 uK (about 2 % dropped at 0.3 depth, 5 uK, z pair)
    assert capped and all(d < 1.0 and t_uk > 2.0 for d, _, t_uk, _ in capped)
    assert max(x for *_, x in capped) == pytest.approx(0.02, abs=0.01)


def test_dropped_population_is_reported_at_the_level_cap():
    s = se.default_settings()
    s["glob"]["temperature_uk"] = 5.0
    for ax in "xyz":
        s["pairs"][ax]["depth"] = 0.3
    res = se.compute(s)
    assert res["n_levels"] == se.MAX_LEVELS
    sp = res["sp"]
    z = sp["axes"]["z"]
    assert 0.01 < z["dropped_population"] < 0.05                 # ~3 % above level 60
    assert sp["dropped_population"] > z["dropped_population"]
    assert "above level 60" in se.level_cut_note(res)
    # an independent count: the Boltzmann sum over 110 computed levels of the z pair (2.97 %)
    from dataclasses import replace
    from lattice_sim.constants import H as h, KB as kb
    from lattice_sim.spectra.sidebands import UPPER, axis_levels
    site = se.site_from(s)
    cfg = [c for c in site.axes if c.axis == "z"][0]
    e = axis_levels(cfg, site.b_lab_gauss, 110)[UPPER]["energies_hz"]
    w = np.exp(-h * (e - e[0]) / (kb * 5e-6))
    direct = w[60:].sum() / w.sum()
    assert z["dropped_population"] == pytest.approx(direct, rel=0.1)
    assert se.level_cut_note(se.compute(se.default_settings())) == ""   # nominal: nothing to report


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
