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
    site = nominal_site()
    cache = {}
    first = spectrum(site, max_dn=1, cache=cache)
    assert len(cache) == 3
    plain = spectrum(site, max_dn=1)
    assert first["carrier_shift_hz"] == pytest.approx(plain["carrier_shift_hz"], abs=1e-6)
    assert np.allclose([l[2:] for l in first["lines"]], [l[2:] for l in plain["lines"]], rtol=1e-9, atol=1e-12)
    warmer = spectrum(replace(site, temperature_k=2e-6), max_dn=1, cache=cache)
    assert len(cache) == 3                               # temperature reuses every pair's levels
    again = spectrum(replace(site, temperature_k=2e-6), max_dn=1)["lines"]
    assert np.allclose([l[2:] for l in warmer["lines"]], [l[2:] for l in again], rtol=1e-9, atol=1e-12)
    spectrum(with_alignment(site, "x", rotating_offset_m=10e-6), max_dn=1, cache=cache)
    assert len(cache) == 4                               # one pair changed, one new entry


# spectrum() of the committed same-site model (e24b39b) at the nominal settings with 13 levels:
# carrier shift, and per pair f(|4,-4>), f(|3,-3>), displacement, sideband / carrier for dn = +1, -1
COMMITTED_NOMINAL_13 = dict(
    carrier_shift_hz=-903.0699712707428,
    x=(18938.062857300727, 18952.530947299827, -3.0570073898661374e-08, 0.19315436014100407, 0.07796746901259613),
    y=(15925.810252182582, 15973.139854730365, 3.645377298910677e-08, 0.25547019658960823, 0.11984951212492896),
    z=(12439.843271391092, 12474.053848631444, 3.351985202690597e-08, 0.20073052019868917, 0.11145748863560749),
)


def test_small_angle_spectrum_equals_the_committed_same_site_model():
    """The large-angle model (tracked wells, neighbouring sites, adiabatic labels, eigenvalue-only
    diagonalization) leaves the nominal spectrum unchanged: rounding of the 9 GHz eigenvalues only."""
    from dataclasses import replace
    sp = spectrum(replace(nominal_site(), n_levels=13), max_dn=1)
    assert sp["carrier_shift_hz"] == pytest.approx(COMMITTED_NOMINAL_13["carrier_shift_hz"], abs=1e-5)
    lines = collapse_lines(sp["lines"])
    for ax in "xyz":
        f_u, f_l, disp, plus, minus = COMMITTED_NOMINAL_13[ax]
        a = sp["axes"][ax]
        assert a["f_upper_hz"] == pytest.approx(f_u, abs=1e-5)
        assert a["f_lower_hz"] == pytest.approx(f_l, abs=1e-5)
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
