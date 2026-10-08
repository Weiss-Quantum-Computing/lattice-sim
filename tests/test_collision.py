"""Tests for the two-atom collision model."""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from lattice_sim.constants import A0
from lattice_sim.gates.collision import (MU, CollisionModel, channel_projectors, collision_ramp, dis_channel_lengths,
                                         interaction_hz, overlap_density, oscillator_width,
                                         scattering_operator, uniform_lengths)

REF = Path(__file__).resolve().parents[1] / "data" / "reference"


def _calibration(name):
    with open(REF / f"calibration_{name}_10kHz_ER316.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fr = np.array([float(r["polarization_rotation_over_pi"]) for r in rows])
    x = np.abs([float(r["minimum_position_continuous_um"]) for r in rows]) * 1e-6
    f = np.array([float(r["trap_frequency_curvature_kHz"]) for r in rows]) * 1e3
    return CubicSpline(np.pi * fr, x), CubicSpline(np.pi * fr, f)


def test_reproduces_transport_optimizer_overlap_tool():
    (x3, f3), (x4, f4) = _calibration("F3"), _calibration("F4")
    a = float(x3(np.pi) + x4(np.pi))
    v = np.linspace(0, 5400, 54001)
    th = np.pi * v / 5400
    o = overlap_density(a - x3(th) - x4(th), f3(th), f4(th), 7e3, 7e3)
    rel = o / o[-1]
    for level, volts in ((0.1, 4340), (0.5, 4785), (0.9, 5155)):
        assert v[np.argmax(rel >= level)] == pytest.approx(volts, abs=2)
    assert interaction_hz(2800 * A0, o[-1]) / 1e3 == pytest.approx(9.48, abs=0.01)


def test_gaussian_overlap_matches_quadrature():
    from scipy.integrate import trapezoid
    f1, f2, d, ft = 9e3, 4e3, 120e-9, 5e3
    s1, s2 = oscillator_width(f1), oscillator_width(f2)
    x = np.linspace(-2e-6, 2e-6, 200001)
    n1 = np.exp(-x**2 / s1**2) / np.sqrt(np.pi * s1**2)
    n2 = np.exp(-(x - d) ** 2 / s2**2) / np.sqrt(np.pi * s2**2)
    numeric_x = trapezoid(n1 * n2, x)
    transverse = (1 / (np.sqrt(2 * np.pi) * oscillator_width(ft))) ** 2
    assert overlap_density(d, f1, f2, ft, ft) / transverse == pytest.approx(numeric_x, rel=1e-6)


def test_dis_stretched_channel_singlet_fraction():
    # |3,3>|3,3> has electronic singlet probability 7/64
    a = dis_channel_lengths(a_s=1.0, a_t=0.0)
    assert a[6] == pytest.approx(7 / 64, abs=1e-12)


def test_channel_projectors_complete():
    total = sum(channel_projectors().values())
    diag = np.real(np.diag(total))
    assert diag == pytest.approx([1, 1, 1, 2, 1, 1, 1], abs=1e-12)


def test_uniform_scattering_has_no_spin_exchange():
    op = scattering_operator(uniform_lengths(2800.0))
    i3 = list(MU).index(3)
    others = [j for j in range(7) if MU[j] not in (3, -3)]
    assert np.max(np.abs(op[i3, others])) < 1e-12 * np.abs(op[i3, i3])


def test_phase_equals_integrated_energy_when_gaps_are_large():
    th = np.linspace(0, np.pi, 50)
    lat = np.full((th.size, 7), 1e9)                 # all other channels far detuned
    lat[:, list(MU).index(3)] = 0.0
    cm = CollisionModel(3, th, 1.2e-6 * th / np.pi * 2, 1.2e-6 * th / np.pi * 2,
                        np.full(th.size, 5e3), np.full(th.size, 5e3), lat)
    op = scattering_operator(uniform_lengths(2800.0))
    t = np.array([0.0, 300.0])
    theta = np.array([np.pi, np.pi])
    phi, pops = cm.evolve(t, theta, op, dt_us=1.0)
    expected = 2 * np.pi * interaction_hz(1.0, cm.overlap(np.pi)) * np.real(op[cm.index, cm.index]) * 300e-6
    assert phi == pytest.approx(expected, rel=1e-3)
    assert pops[cm.index] == pytest.approx(1.0, abs=1e-6)


def test_collision_ramp_times_are_strictly_increasing():
    t = np.linspace(0, 100, 81)
    theta = np.pi * t / t[-1]
    for hold in (0.0, 50.0):
        tr, thr = collision_ramp(t, theta, 0.6 * np.pi, hold)
        assert np.all(np.diff(tr) > 0), f"hold {hold}: duplicated time points"
        assert tr.size == thr.size
        assert thr.max() == pytest.approx(0.6 * np.pi)
        assert tr[-1] == pytest.approx(2 * t[np.searchsorted(theta, 0.6 * np.pi)] + hold)


def test_evolve_integrates_the_whole_span():
    # a span that is not a multiple of dt_us must not lose its last fraction
    th = np.linspace(0, np.pi, 50)
    lat = np.full((th.size, 7), 1e9)
    lat[:, list(MU).index(3)] = 0.0
    cm = CollisionModel(3, th, 1.2e-6 * th / np.pi * 2, 1.2e-6 * th / np.pi * 2,
                        np.full(th.size, 5e3), np.full(th.size, 5e3), lat)
    op = scattering_operator(uniform_lengths(2800.0))
    u_hz = interaction_hz(1.0, cm.overlap(np.pi)) * np.real(op[cm.index, cm.index])
    for span in (300.0, 300.7):
        phi, _ = cm.evolve(np.array([0.0, span]), np.array([np.pi, np.pi]), op, dt_us=1.0)
        # 1e-5 of second-order shift from the far channels; the old code lost 2.3e-3 at 300.7
        assert phi == pytest.approx(2 * np.pi * u_hz * span * 1e-6, rel=1e-4)
