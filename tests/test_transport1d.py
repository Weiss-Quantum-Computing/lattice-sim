"""Acceptance tests for the 1D transport engines."""

import numpy as np
import pytest

from lattice_sim.constants import HBAR, MASS_CS
from lattice_sim.motion.gaussian_exact import propagate_gaussian
from lattice_sim.motion.transport1d import PotentialTable, Transport1D, uniform_grid

THETAS = np.linspace(0, np.pi, 2881)  # theta interpolation sets the error floor
X = uniform_grid(-1.0e-6, 3.0e-6, 512)


def centre(theta):
    return 2.4e-6 * (1 - np.cos(theta)) / 2


def frequency(theta):
    return 10e3 - 6e3 * np.sin(theta) ** 2


def test_sudden_displacement_gives_coherent_state():
    # Trap jumps by d at t = 0: P0 = exp(-|alpha|^2), |alpha|^2 = m omega d^2 / 2 hbar
    d, f = 50e-9, 10e3
    table = PotentialTable.harmonic(lambda th: d * (th > 0), lambda th: f + 0 * th,
                                    np.array([0.0, 1e-9, 1.0]), X)
    tr = Transport1D(table)
    psi0, _ = tr.ground_state(0.0)
    _, p0, _ = tr.propagate([0, 50], [1.0, 1.0], dt_us=0.1, psi0=psi0)
    expected = np.exp(-MASS_CS * 2 * np.pi * f * d**2 / (2 * HBAR))
    assert p0[-1] == pytest.approx(expected, abs=2e-5)


def test_split_operator_matches_exact_gaussian():
    t = np.linspace(0, 1500, 1501)
    theta = np.pi * (t / t[-1] - np.sin(2 * np.pi * t / t[-1]) / (2 * np.pi))
    table = PotentialTable.harmonic(centre, frequency, THETAS, X)
    ts, p_split, _ = Transport1D(table).propagate(t, theta, dt_us=0.1, sample_every_us=20)
    _, p_exact, _ = propagate_gaussian(t, theta, centre, frequency, sample_times_us=ts)
    assert np.max(np.abs(p_split - p_exact)) < 2e-5
    assert p_exact.min() < 0.99  # the test ramp is fast enough to excite
