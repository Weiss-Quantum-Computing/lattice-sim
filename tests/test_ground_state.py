"""Acceptance tests for the ground-state module.

Targets are the values in the Lattice Beams & Magnetic Fields parameter
reference (Sec. 4.4), which were themselves checked against the theses.
"""

import numpy as np
import pytest

from lattice_sim.atom.ground_state import (
    breit_rabi,
    eigenlevels,
    field_sensitivity,
    magic_field,
    transition_frequency,
)
from lattice_sim.constants import NU_HFS_HZ

QUBIT = ((4, 1), (3, -1))


def test_zero_field_splitting():
    split = breit_rabi(4, 0, 0.0) - breit_rabi(3, 0, 0.0)
    assert split == pytest.approx(NU_HFS_HZ, rel=1e-12)


@pytest.mark.parametrize(
    "direction",
    [(0, 0, 1), (1, 1, 0), (0.3, -0.7, 0.2)],
)
def test_matrix_matches_breit_rabi_any_direction(direction):
    b = 1.3924
    axis = np.asarray(direction, float) / np.linalg.norm(direction)
    levels = eigenlevels(b * axis)
    for (f, m), (energy, _) in levels.items():
        assert energy == pytest.approx(breit_rabi(f, m, b), abs=1e-3)


def test_magic_field():
    # Breit-Rabi and the full 16 x 16 Hamiltonian both give 1.39304 G; the
    # lowest-order closed form is B = -32 g_I h nu_hfs / (15 (g_J - g_I)^2 mu_B).
    # The theses quote 1.393 G. The parameter reference's "1.3924 G" is
    # 0.6 mG low; the qubit frequency differs by < 1 mHz between the two.
    assert magic_field(*QUBIT) == pytest.approx(1.39304, abs=2e-5)


def test_qubit_sensitivity_near_magic():
    b0 = magic_field(*QUBIT)
    for b in (b0 - 0.1, b0 + 0.1):
        assert abs(field_sensitivity(*QUBIT, b)) / 1e3 == pytest.approx(0.08, abs=0.01)


@pytest.mark.parametrize("b, expected", [(0.8, -0.48), (0.9, -0.40), (1.0, -0.31)])
def test_qubit_sensitivity_low_field(b, expected):
    assert field_sensitivity(*QUBIT, b) / 1e3 == pytest.approx(expected, abs=0.01)


def test_stretched_transition_sensitivity():
    # |4,-4> -> |3,-3>: -2.452 kHz/mG
    sensitivity = field_sensitivity((4, -4), (3, -3), 1.3924) / 1e3 / 1e3
    assert sensitivity == pytest.approx(-2.452, abs=0.002)


def test_rf_splittings_at_magic_field():
    b = 1.3924
    # In F = 3 (g_F < 0) |3,-1> lies above |3,0>, so compare magnitudes.
    assert abs(transition_frequency((3, 0), (3, -1), b)) / 1e3 == pytest.approx(488.7, abs=0.1)
    assert abs(transition_frequency((4, 1), (4, 0), b)) / 1e3 == pytest.approx(487.1, abs=0.1)
