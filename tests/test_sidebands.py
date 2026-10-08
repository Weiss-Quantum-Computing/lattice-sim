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
