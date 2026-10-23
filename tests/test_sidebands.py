"""Tests for the microwave carrier / sideband model."""

import numpy as np
import pytest

from lattice_sim.constants import H, HBAR, MASS_CS
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.spectra.sidebands import (AxisConfig, SiteConfig, b_in_pair_frame, collapse_lines,
                                           franck_condon, nominal_site, spectrum, thermal_weights,
                                           vibrational_levels, with_alignment)


def harmonic(x, f_hz, x0=0.0):
    return 0.5 * MASS_CS * (2 * np.pi * f_hz) ** 2 * (x - x0) ** 2 / H


X = np.linspace(-1.5e-6, 1.5e-6, 1201)


def test_vibrational_levels_of_a_harmonic_trap():
    e, psi = vibrational_levels(X, harmonic(X, 10e3), n_levels=4)
    assert np.diff(e) == pytest.approx([10e3] * 3, rel=1e-4)
    assert e[0] == pytest.approx(5e3, rel=1e-4)
    assert np.allclose(np.sum(psi**2, axis=1), 1.0)


def test_franck_condon_of_a_displaced_trap():
    # |<1|0>|^2 = d^2 / (4 x_zp^2) with x_zp^2 = hbar / (2 m omega), to first order
    f, d = 10e3, 20e-9
    _, psi_a = vibrational_levels(X, harmonic(X, f), 3)
    _, psi_b = vibrational_levels(X, harmonic(X, f, d), 3)
    fc = franck_condon(psi_a, psi_b)
    eta2 = d**2 / (4 * HBAR / (2 * MASS_CS * 2 * np.pi * f))
    assert fc[1, 0] == pytest.approx(eta2 * np.exp(-eta2), rel=2e-3)
    assert fc[0, 0] == pytest.approx(np.exp(-eta2), rel=1e-4)
    assert np.allclose(fc.sum(axis=0)[:1], 1.0, atol=1e-6)


def test_identical_potentials_give_no_sidebands():
    _, psi = vibrational_levels(X, harmonic(X, 10e3), 3)
    assert np.allclose(franck_condon(psi, psi), np.eye(3), atol=1e-10)


def test_thermal_weights():
    assert thermal_weights(10e3, 0.0, 3).tolist() == [1.0, 0.0, 0.0]
    w = thermal_weights(10e3, 5e-6, 20)
    assert w[1] / w[0] == pytest.approx(np.exp(-H * 10e3 / (1.380649e-23 * 5e-6)))


def test_pair_frames_are_right_handed_and_map_the_bias_field():
    from lattice_sim.spectra.sidebands import PAIR_FRAMES
    for frame in PAIR_FRAMES.values():
        assert np.linalg.det(frame) == pytest.approx(1.0)
        assert np.allclose(frame @ frame.T, np.eye(3))
    b = np.array([1.0, 1.0, 0.0])
    assert b_in_pair_frame("x", b).tolist() == [1.0, 1.0, 0.0]
    assert b_in_pair_frame("y", b).tolist() == [1.0, -1.0, 0.0]
    assert b_in_pair_frame("z", b).tolist() == [0.0, -1.0, 1.0]


def test_pure_linear_unrotated_lattice_has_no_sidebands_and_the_zero_point_shift():
    site = nominal_site(rotation_fraction=0.0, extinction_ratio=np.inf)
    sp = spectrum(site, max_dn=1)
    strongest = max(w for *_, w in sp["lines"]) if sp["lines"] else 0.0
    assert strongest < 1e-8
    # the carrier shift is the hyperfine differential of the zero-point
    # energy: -0.226 % of sum_axes h f / 4 (|4,-4> is shifted less)
    zp = sum(cfg.trap_frequency_hz for cfg in site.axes) / 4
    assert sp["carrier_shift_hz"] == pytest.approx(-0.00226 * zp, rel=0.05)
    for ax, cfg in zip("xyz", site.axes):
        assert sp["axes"][ax]["f_lower_hz"] == pytest.approx(cfg.trap_frequency_hz, rel=2e-3)


def test_rotation_displaces_the_wells_and_makes_sidebands():
    site = nominal_site(rotation_fraction=0.03, extinction_ratio=np.inf)
    sp = spectrum(site, max_dn=1)
    x = sp["axes"]["x"]
    assert abs(x["displacement_m"]) > 5e-9
    lines = collapse_lines(sp["lines"], "x")
    assert lines[("x", 1)][1] > 1e-3            # one quantum added, near -f_lower
    assert lines[("x", 1)][0] == pytest.approx(-x["f_lower_hz"], rel=5e-3)
    # the n = 1 -> n' = 0 line sits exactly at +f_upper (other n differ by the anharmonicity)
    minus = [off for ax, dn, off, _ in sp["lines"] if ax == "x" and dn == -1]
    assert min(abs(off - x["f_upper_hz"]) for off in minus) < 1e-6 * x["f_upper_hz"]


def test_centering_offset_lowers_the_trap_frequency():
    site = nominal_site(rotation_fraction=0.0, extinction_ratio=np.inf)
    off = with_alignment(site, "x", static_offset_m=30e-6, rotating_offset_m=30e-6)
    f0 = spectrum(site, max_dn=1)["axes"]["x"]["f_lower_hz"]
    f1 = spectrum(off, max_dn=1)["axes"]["x"]["f_lower_hz"]
    # intensity falls by exp(-2 r^2 / w^2) = exp(-0.5), frequency by exp(-0.25)
    assert f1 / f0 == pytest.approx(np.exp(-0.25), rel=2e-3)


def test_beam_geometry_nominal_and_tilts():
    from lattice_sim.spectra.beams import ArmGeometry, PairGeometry, NOMINAL_HALF_ANGLE
    nominal = PairGeometry(ArmGeometry(), ArmGeometry())
    spacing, n, frame = nominal.lattice()
    assert spacing == pytest.approx(4.8374e-6, rel=1e-6)
    assert np.allclose(n, [1, 0, 0]) and np.allclose(frame, np.eye(3))
    assert nominal.amplitudes(np.zeros(3)) == pytest.approx([1.0, 1.0])
    # both arms offset by r: amplitude exp(-r^2/w^2) at the centre, 1 at the beam axis
    off = PairGeometry(ArmGeometry(offset_pol_m=30e-6), ArmGeometry(offset_pol_m=30e-6))
    assert off.amplitudes(np.zeros(3)) == pytest.approx([np.exp(-0.25)] * 2)
    assert off.amplitudes(np.array([0, 0, 30e-6])) == pytest.approx([1.0, 1.0])
    # differential in-plane tilt changes the half angle and the spacing
    d = 1e-3
    cross = PairGeometry(ArmGeometry(tilt_inplane_rad=d), ArmGeometry(tilt_inplane_rad=-d))
    assert cross.lattice()[0] == pytest.approx(843.222e-9 / (2 * np.sin(NOMINAL_HALF_ANGLE + d)), rel=1e-9)
    # differential vertical tilt rotates the fringe normal by ~ d / sin(alpha)
    vert = PairGeometry(ArmGeometry(tilt_vertical_rad=d), ArmGeometry(tilt_vertical_rad=-d))
    _, n_v, _ = vert.lattice()
    assert np.arctan2(n_v[2], n_v[0]) == pytest.approx(d / np.sin(NOMINAL_HALF_ANGLE), rel=1e-3)
    # a common tilt leaves spacing and normal unchanged; a pivot moves the beams
    common = PairGeometry(ArmGeometry(tilt_vertical_rad=d, pivot_m=0.3), ArmGeometry(tilt_vertical_rad=d, pivot_m=0.3))
    assert common.lattice()[0] == pytest.approx(spacing, rel=1e-6)
    assert np.allclose(common.lattice()[1], n, atol=1e-9)
    assert common.amplitudes(np.array([0, 0, 0.3 * d])) == pytest.approx([1.0, 1.0], abs=1e-9)


def test_crossing_angle_scales_the_trap_frequency_at_fixed_depth():
    site = nominal_site(rotation_fraction=0.0, extinction_ratio=np.inf)
    a0 = site.axes[0].spacing_m
    half = np.rad2deg(np.arcsin(843.222e-9 / (2 * a0)))
    f0 = spectrum(site, max_dn=1)["axes"]["x"]["f_lower_hz"]
    f1 = spectrum(with_alignment(site, "x", crossing_half_angle_deg=half + 0.1), max_dn=1)["axes"]["x"]["f_lower_hz"]
    a1 = 843.222e-9 / (2 * np.sin(np.deg2rad(half + 0.1)))
    assert f1 / f0 == pytest.approx(a0 / a1, rel=2e-3)


def _nominal_x_potentials(theta=0.03 * np.pi):
    from lattice_sim.fields.potential import StatePotential
    from lattice_sim.spectra.sidebands import LOWER, UPPER
    site = nominal_site()
    cfg = site.axes[0]
    model = StatePotential(cfg.pair(), cfg.b_local(site.b_lab_gauss), trap_frequency_hz=cfg.trap_frequency_hz)
    return model, model.potentials((UPPER, LOWER), X, theta)


def test_sparse_and_banded_solvers_agree_on_the_lattice_wells():
    from lattice_sim.spectra.sidebands import LOWER, UPPER, vibrational_levels_banded
    _, pots = _nominal_x_potentials()
    psis = {}
    for state in (UPPER, LOWER):
        e, psi = vibrational_levels(X, pots[state], 13)
        e_ref, psi_ref = vibrational_levels_banded(X, pots[state], 13)
        assert np.max(np.abs(e - e_ref)) < 1e-6          # Hz, on ~20 kHz level spacings
        assert np.allclose(np.abs(np.sum(psi * psi_ref, axis=1)), 1.0, atol=1e-9)
        psis[state] = (psi, psi_ref)
    fc = franck_condon(psis[UPPER][0], psis[LOWER][0])
    fc_ref = franck_condon(psis[UPPER][1], psis[LOWER][1])
    assert np.max(np.abs(fc - fc_ref)) < 1e-9


def test_shared_diagonalization_matches_the_single_state_potentials():
    from lattice_sim.spectra.sidebands import LOWER, UPPER
    model, pots = _nominal_x_potentials()
    for state in (UPPER, LOWER):
        assert np.array_equal(pots[state], model.potential(state, X, 0.03 * np.pi))


def test_spectrum_cache_reuses_levels_and_changes_nothing():
    from dataclasses import replace
    site = replace(nominal_site(), n_levels=13)          # a fixed level count: temperature is free
    cache = {}
    first = spectrum(site, max_dn=1, cache=cache)
    assert len(cache) == 3
    plain = spectrum(site, max_dn=1)
    assert first["carrier_shift_hz"] == pytest.approx(plain["carrier_shift_hz"], abs=1e-5)
    assert np.allclose([l[2:] for l in first["lines"]], [l[2:] for l in plain["lines"]], rtol=1e-9, atol=1e-12)
    warmer = spectrum(replace(site, temperature_k=2e-6), max_dn=1, cache=cache)
    assert len(cache) == 3                               # temperature reuses every pair's levels
    again = spectrum(replace(site, temperature_k=2e-6), max_dn=1)["lines"]
    assert np.allclose([l[2:] for l in warmer["lines"]], [l[2:] for l in again], rtol=1e-9, atol=1e-12)
    spectrum(with_alignment(site, "x", rotating_offset_m=10e-6), max_dn=1, cache=cache)
    assert len(cache) == 4                               # one pair changed, one new entry
    # the level count from the temperature (n_levels None): 2 uK needs more levels, so it recomputes
    auto, cache = nominal_site(), {}
    spectrum(auto, max_dn=1, cache=cache)
    spectrum(replace(auto, temperature_k=2e-6), max_dn=1, cache=cache)
    assert len(cache) == 6


def test_measured_extinction_profile_is_cached_by_its_values():
    from dataclasses import replace
    from lattice_sim.fields.extinction import ExtinctionProfile
    prof = ExtinctionProfile(np.array([0.0, np.pi]), np.array([3e4, 2e3]), np.array([False, False]))
    site = nominal_site()
    site = replace(site, axes=[replace(c, extinction_ratio=prof) for c in site.axes])
    cache = {}
    first = spectrum(site, max_dn=1, cache=cache)
    assert len(cache) == 3                               # was not cached at all (callable)
    same = ExtinctionProfile(prof.theta.copy(), prof.er.copy(), prof.lower.copy())
    again = spectrum(replace(site, axes=[replace(c, extinction_ratio=same) for c in site.axes]), max_dn=1,
                     cache=cache)
    assert len(cache) == 3 and again["carrier_shift_hz"] == first["carrier_shift_hz"]
    other = ExtinctionProfile(prof.theta.copy(), np.array([3e4, 1e3]), prof.lower.copy())
    spectrum(replace(site, axes=[replace(c, extinction_ratio=other) for c in site.axes]), max_dn=1, cache=cache)
    assert len(cache) == 6                               # different values: new levels
    lam = replace(site, axes=[replace(c, extinction_ratio=lambda th: 3e4) for c in site.axes])
    spectrum(lam, max_dn=1, cache=cache)
    assert len(cache) == 6                               # another callable: not cached


def test_thermal_weights_use_the_computed_level_energies():
    from lattice_sim.constants import KB
    from lattice_sim.spectra.sidebands import (dropped_population, thermal_weights_from_energies)
    e = np.array([0.0, 10e3, 19.5e3, 28.5e3])             # an anharmonic ladder (Hz)
    w = thermal_weights_from_energies(e + 123.0, 2e-6)
    beta = H / (KB * 2e-6)
    assert w == pytest.approx(np.exp(-beta * e) / np.sum(np.exp(-beta * e)), rel=1e-12)
    assert thermal_weights_from_energies(e, 0.0).tolist() == [1.0, 0.0, 0.0, 0.0]
    # a harmonic ladder: the estimate of the population above the last level is exact (r^n)
    f, n, temp = 10e3, 8, 3e-6
    r = np.exp(-H * f / (KB * temp))
    assert dropped_population(f * np.arange(n), temp) == pytest.approx(r ** n, rel=1e-9)
    sp = spectrum(nominal_site(), max_dn=1)
    x = sp["axes"]["x"]
    e_u = x["levels"][(4, -4)]["energies_hz"]
    assert x["thermal"] == pytest.approx(thermal_weights_from_energies(e_u, 1e-6), rel=1e-12)


def test_level_count_follows_the_temperature():
    """SiteConfig.n_levels None (the default) keeps <= 1e-3 of the thermal population above the
    last level; the old fixed 6 levels put the x sideband / carrier ratio 45 % low at 5 uK."""
    from dataclasses import replace
    from lattice_sim.spectra.sidebands import DROPPED_POPULATION, MAX_LEVELS, n_levels_for, site_n_levels
    assert site_n_levels(nominal_site()) == 13 == n_levels_for(1e-6, [19e3, 16e3, 12.5e3])
    assert n_levels_for(0.0, [12.5e3]) == 6 and n_levels_for(30e-6, [5e3]) == MAX_LEVELS
    warm = replace(nominal_site(), temperature_k=5e-6)
    n = site_n_levels(warm)
    auto = spectrum(warm, max_dn=1)
    assert auto["n_levels"] == n and auto["dropped_population"] <= 3 * DROPPED_POPULATION
    more = spectrum(replace(warm, n_levels=n + 15), max_dn=1)
    six = spectrum(replace(warm, n_levels=6), max_dn=1)
    ratio = {k: s["axes"]["x"]["dn_weights"][1] / s["axes"]["x"]["dn_weights"][0]
             for k, s in (("auto", auto), ("more", more), ("six", six))}
    assert ratio["auto"] == pytest.approx(ratio["more"], rel=5e-3)
    assert ratio["six"] < 0.7 * ratio["auto"]                       # the old default's bias
    assert six["dropped_population"] > 0.1


def test_eigen_window_grows_for_high_levels_of_a_shallow_trap():
    """z pair at 0.3 depth with 60 levels: the fixed +-1.2 um window moved E_59 by 4.8 kHz; the
    window now grows until the top level is converged (equal to a window twice as large)."""
    from dataclasses import replace
    from lattice_sim.spectra.sidebands import (LEVEL_WINDOW_M, UPPER, WINDOW_EDGE_LIMIT, axis_levels)
    site = nominal_site()
    z = replace([c for c in site.axes if c.axis == "z"][0], trap_frequency_hz=12.5e3 * np.sqrt(0.3))
    auto = axis_levels(z, site.b_lab_gauss, 60)
    fixed = axis_levels(z, site.b_lab_gauss, 60, window_m=LEVEL_WINDOW_M)
    wide = axis_levels(z, site.b_lab_gauss, 60, window_m=0.45 * z.spacing_m)      # the size cap, 2.18 um
    assert auto[UPPER]["window_m"] > LEVEL_WINDOW_M
    assert auto[UPPER]["edge_fraction"] <= WINDOW_EDGE_LIMIT < fixed[UPPER]["edge_fraction"]
    e_auto, e_wide = auto[UPPER]["energies_hz"], wide[UPPER]["energies_hz"]
    assert np.max(np.abs(e_auto - e_wide)) < 1e-3                  # Hz
    assert fixed[UPPER]["energies_hz"][59] - e_wide[59] > 1e3       # the hard wall raised level 59
    # where the default window suffices nothing changes
    nominal = axis_levels(site.axes[0], site.b_lab_gauss, 13)
    assert nominal[UPPER]["window_m"] == LEVEL_WINDOW_M


# spectrum() of the committed same-site model (e24b39b) at the nominal settings with 13 levels:
# carrier shift, and per pair f(|4,-4>), f(|3,-3>), displacement, sideband / carrier for dn = +1, -1.
# The sideband / carrier ratios are re-pinned for the thermal weights from the computed (anharmonic)
# level energies (fixes 2026-10-15, G5-27); e24b39b's harmonic weights gave x 0.19315436014100407 /
# 0.07796746901259613, y 0.25547019658960823 / 0.11984951212492896, z 0.20073052019868917 /
# 0.11145748863560749. A regression pin of this code's own output, not an independent reference.
COMMITTED_NOMINAL_13 = dict(
    carrier_shift_hz=-903.0699712707428,
    x=(18938.062857300727, 18952.530947299827, -3.0570073898661374e-08, 0.19326072315186937, 0.07807708878268207),
    y=(15925.810252182582, 15973.139854730365, 3.645377298910677e-08, 0.2556805949147516, 0.12007357618097064),
    z=(12439.843271391092, 12474.053848631444, 3.351985202690597e-08, 0.2010717523937422, 0.11182130403275019),
)


def test_small_angle_spectrum_equals_the_committed_same_site_model():
    """The large-angle model (tracked wells, neighbouring sites, adiabatic labels, eigenvalue-only
    diagonalization) leaves the nominal spectrum unchanged: rounding of the 9 GHz eigenvalues only.
    A regression pin (the e24b39b output), not a physics check."""
    from dataclasses import replace
    sp = spectrum(replace(nominal_site(), n_levels=13), max_dn=1)
    # 1e-4 Hz: the levels are eigenvalues of a 16 x 16 Hamiltonian with 9.19 GHz entries (float64
    # resolution ~1e-6 Hz each), and threaded LAPACK differs run to run (Windows CI: 1.24e-5 Hz)
    tol_hz = 1e-4
    assert sp["carrier_shift_hz"] == pytest.approx(COMMITTED_NOMINAL_13["carrier_shift_hz"], abs=tol_hz)
    lines = collapse_lines(sp["lines"])
    for ax in "xyz":
        f_u, f_l, disp, plus, minus = COMMITTED_NOMINAL_13[ax]
        a = sp["axes"][ax]
        assert a["f_upper_hz"] == pytest.approx(f_u, abs=tol_hz)
        assert a["f_lower_hz"] == pytest.approx(f_l, abs=tol_hz)
        assert a["displacement_m"] == pytest.approx(disp, abs=1e-14)
        assert lines[(ax, 1)][1] / sp["carrier_weight"] == pytest.approx(plus, rel=1e-8)
        assert lines[(ax, -1)][1] / sp["carrier_weight"] == pytest.approx(minus, rel=1e-8)
        assert a["dn_weights"][0] == pytest.approx(a["diagonal"], rel=1e-12)
        assert a["dn_weights"][1] / a["dn_weights"][0] == pytest.approx(plus, rel=1e-8)   # per-pair ratio
        assert a["dn_weights"][-1] / a["dn_weights"][0] == pytest.approx(minus, rel=1e-8)
        assert list(a["diagonal_sites"]) == [0]                 # no neighbouring-site well within reach
        assert a["levels"][(4, -4)]["x"].size == 1201           # the default grid
    assert sp["carrier_weight_same_site"] == sp["carrier_weight"]
    assert [l[:2] for l in sp["site_lines"]] == [(l[0], 0) for l in sp["lines"]]
