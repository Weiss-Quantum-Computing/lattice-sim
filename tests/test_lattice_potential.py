"""Acceptance tests for the light shift and the x-pair lattice potential."""

import numpy as np
import pytest

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.atom.polarizability import (
    differential_scalar_ratio,
    effective_hyperfine_estimate,
)
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

B_Q = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])


def test_differential_light_shift_matches_closed_form():
    lam = 843.222e-9
    assert differential_scalar_ratio(lam) == pytest.approx(
        effective_hyperfine_estimate(lam), rel=0.02
    )


def test_differential_shift_per_level_matches_wu():
    # Wu (2020) p.25: ~28-30 Hz per vibrational level at 847 nm and 15 kHz.
    # Per level the mean potential energy rises by h f / 2.
    per_level = abs(differential_scalar_ratio(847.036e-9)) * 15e3 / 2
    assert 28 <= per_level <= 31


@pytest.mark.parametrize("state", [(3, -3), (4, -3), (3, 1), (3, -1)])
def test_pure_linear_lattice_is_state_independent_at_theta_zero(state):
    model = StatePotential(LatticePair(), B_Q, include_hyperfine_scalar=False)
    x_min, f = model.minimum(state, 0.0)
    assert x_min == pytest.approx(0.0, abs=1e-10)
    assert f == pytest.approx(10e3, rel=1e-3)


@pytest.mark.parametrize("state, sign", [((3, -3), -1), ((4, -3), 1)])
def test_half_site_displacement_at_theta_pi(state, sign):
    pair = LatticePair()
    model = StatePotential(pair, B_Q)
    x, _ = model.track(state, np.linspace(0, np.pi, 25))
    assert x[-1] == pytest.approx(sign * pair.spacing_m / 2, abs=1e-9)


def test_extinction_ratio_316_reproduces_calibration_split():
    # Transport-Optimizer calibration_{F3,F4}_10kHz_ER316.csv at theta = 0:
    # f(4,-3) - f(3,-3) = 10.026039 - 9.924940 kHz = 101.10 Hz.
    pair = LatticePair(extinction_ratio=316.0, handedness=-1)
    model = StatePotential(pair, B_Q, include_hyperfine_scalar=False)
    _, f3 = model.minimum((3, -3), 0.0)
    _, f4 = model.minimum((4, -3), 0.0)
    assert f4 - f3 == pytest.approx(101.10, abs=1.0)
