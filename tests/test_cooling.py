"""Tests for projection sideband cooling: AFP propagation, optical pumping,
recoil and the 3D cycle."""

import numpy as np
import pytest

from lattice_sim.constants import H, HBAR, MASS_CS
from lattice_sim.cooling import afp, psc
from lattice_sim.cooling.optical_pumping import (decay_branching, harmonic_kick_probability,
                                                 kick_matrices, lamb_dicke, photon_cycle_matrix,
                                                 photon_number_distribution, wigner3j, wigner6j)
from lattice_sim.spectra.sidebands import vibrational_levels

X = np.linspace(-1.5e-6, 1.5e-6, 1201)


def harmonic(x, f_hz, x0=0.0):
    return 0.5 * MASS_CS * (2 * np.pi * f_hz) ** 2 * (x - x0) ** 2 / H


def toy_ladder(axis="x", f=18e3, d=40e-9, n=12, carrier=0.0):
    e_u, psi_u = vibrational_levels(X, harmonic(X, f), n)
    e_l, psi_l = vibrational_levels(X, harmonic(X, f * 1.002, d), n)
    return psc.Ladder(axis, 0.0, e_u - e_u[0], e_l - e_l[0], psi_l @ psi_u.T, carrier, d), psi_u


# ---------------------------------------------------------------- AFP
@pytest.mark.parametrize("overlap,detuning", [(0.05, 0.0), (0.1, 0.0), (0.2, 2e3), (1.0, -3e3)])
def test_piecewise_afp_matches_ode(overlap, detuning):
    """8 us piecewise-constant steps against the smooth waveform (adaptive ODE)."""
    a = afp.two_level_transfer(overlap, detuning, -4e3, 20e3)
    b = afp.two_level_transfer_ode(overlap, detuning, -4e3, 20e3)
    assert a == pytest.approx(b, abs=2e-4)


def test_afp_sweep_direction_does_not_matter_for_two_levels():
    assert afp.two_level_transfer(0.08, 1e3, 6e3) == pytest.approx(afp.two_level_transfer(0.08, 1e3, -6e3), abs=1e-6)


def test_afp_misses_a_line_outside_the_sweep():
    assert afp.two_level_transfer(1.0, 9e3, -6e3) < 1e-3


def test_ladder_pulse_is_unitary_and_drives_the_sideband():
    lad, _ = toy_ladder()
    t = psc.pulse_matrix(lad, 1, -4e3, 20e3, center_hz=lad.e_u[1] - lad.e_l[0])
    n = lad.e_u.size
    assert np.allclose(t.sum(axis=0), 1.0, atol=1e-10)
    # |4,-4,1> -> |3,-3,0> and |3,-3,0> -> |4,-4,1>
    assert t[n + 0, 1] > 0.95 and t[1, n + 0] > 0.95
    # the ground state |4,-4,0> has no Delta n = -1 partner: it stays
    assert t[0, 0] > 0.99


# ---------------------------------------------------------------- angular algebra, OP
def test_wigner_symbols_known_values():
    assert wigner3j(1, 1, 0, 1, -1, 0) == pytest.approx(1 / np.sqrt(3))
    assert wigner3j(0.5, 0.5, 1, 0.5, 0.5, -1) == pytest.approx(-1 / np.sqrt(3))
    assert wigner6j(1, 1, 1, 1, 1, 1) == pytest.approx(1 / 6)


def test_branching_ratios_of_f_prime_4():
    b = decay_branching(-3)
    assert sum(v for (f, m, q), v in b.items() if f == 4) == pytest.approx(7 / 12)
    assert decay_branching(-4)[(4, -4, 0)] == pytest.approx(7 / 15)


def test_photon_number_from_3_minus_3():
    dist, pi = photon_number_distribution((3, -3))
    k = np.arange(dist.size)
    assert dist.sum() == pytest.approx(1.0, abs=1e-9)
    assert dist[0] == 0.0
    assert k @ dist == pytest.approx(15 / 7, rel=1e-6)   # geometric with p = 7/15
    assert pi == pytest.approx(7 / 15)


def test_kick_matrices_against_analytic_displacement_operator():
    f = 18e3
    _, psi = vibrational_levels(X, harmonic(X, f), 8)
    eta = lamb_dicke(f)
    k = 2 * np.pi / 852.347e-9
    m = kick_matrices(X, psi, [k])[0]
    for (i, j) in [(0, 0), (1, 0), (2, 0), (3, 1), (5, 5), (6, 4)]:
        assert m[i, j] == pytest.approx(harmonic_kick_probability(i, j, eta), abs=2e-5)


def test_one_photon_heating_in_the_lamb_dicke_limit():
    """P(0 -> 1) ~ eta^2 (u_abs^2 + <u_em^2>) for weak kicks."""
    f = 200e3      # deep trap, eta ~ 0.1
    _, psi = vibrational_levels(np.linspace(-0.5e-6, 0.5e-6, 2001), harmonic(np.linspace(-0.5e-6, 0.5e-6, 2001), f), 4)
    m = photon_cycle_matrix(np.linspace(-0.5e-6, 0.5e-6, 2001), psi, "x", pi_fraction=0.0)
    eta2 = lamb_dicke(f) ** 2
    # sigma pattern about (1,1,0)/sqrt2: <u_x^2> = (1/3)(1 - P2 term) -> computed by quadrature in the code
    from lattice_sim.cooling.optical_pumping import emission_projection_weights, OP_BEAM_DIRECTION_LAB
    nodes, w = emission_projection_weights(np.array([1.0, 0, 0]), OP_BEAM_DIRECTION_LAB, 0.0)
    expected = eta2 * (0.5 + w @ nodes**2)
    assert m[1, 0] == pytest.approx(expected, rel=0.03)
    # sigma dipole pattern: <cos^2> = 2/5 along q, 3/10 per transverse axis;
    # the x axis is at 45 deg to q: <u^2> = 0.5 * 2/5 + 0.5 * 3/10 = 0.35
    assert w @ nodes**2 == pytest.approx(0.35, abs=0.005)


# ---------------------------------------------------------------- 3D cycle
def toy_library(n=10):
    ladders = {}
    for axis, f in zip(psc.AXES, (18e3, 15e3, 12.5e3)):
        lad, psi = toy_ladder(axis, f, 45e-9, n)
        ladders[(axis, 0.0)] = lad
        ladders[(axis, 12.0)] = lad
        ladders[(axis, 6.0)] = lad
        ladders[(axis, 13.0)] = lad
        ladders[(axis, "x_grid")] = X
        ladders[(axis, "psi0")] = psi
    lib = psc.StepLibrary(ladders, 20e3, center_mode="line")
    return ladders, lib


def test_factorized_matches_3d_and_probability_is_conserved():
    ladders, lib = toy_library(16)
    op = psc.optical_pumping_model(ladders)
    seq = [psc.Loop(3, dict.fromkeys(psc.AXES, 12.0), dict.fromkeys(psc.AXES, 6.0))]
    p1 = {a: psc.thermal(16, f, 1.5e-6) for a, f in zip(psc.AXES, (18e3, 15e3, 12.5e3))}
    p3 = np.einsum("i,j,k->ijk", p1["x"], p1["y"], p1["z"])
    hist3, _ = psc.simulate_3d(seq, lib, op, p0=p3)
    histf = psc.simulate_factorized(seq, lib, op, p0=p1)
    g3 = hist3[-1]["ground_3d"]
    gf = np.prod([histf[-1][a][0] for a in psc.AXES])
    assert g3 > p3[0, 0, 0]                       # it cools
    assert gf == pytest.approx(g3, abs=0.01)      # mean-field spectators
    assert hist3[-1]["total"] == pytest.approx(1.0, abs=0.02)  # truncation loss only
