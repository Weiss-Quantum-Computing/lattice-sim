"""Tests for the echo-sequence and noise engine."""

import numpy as np
import pytest

from lattice_sim.spin.echo import (LegTransient, NoiseSpectrum, Sequence, Transport, coefficients,
                                   filter_variance, phase, sample_noise)
from lattice_sim.spin.pulses import train_contrast, worst_input_contrast


class FlatPair:
    """Pair physics with a simple analytic differential shift, for tests."""
    kappa_b_hz_per_mg = -2000.0
    storage_ls_hz = 30.0

    def de0(self, th):
        return -1e4 * np.sin(th) ** 2

    def s(self, th):
        return -1e6 * np.sin(th) ** 4

    def ddtheta(self, th, eta):
        return -2e4 * np.sin(th) * np.cos(th) - eta * 4e6 * np.sin(th) ** 3 * np.cos(th)


def transport(axis="x"):
    t = np.linspace(0, 2000, 201)
    return Transport(t, np.pi * np.sin(np.pi / 2 * t / t[-1]) ** 2, 300.0, axis)


def symmetric_echo(gap_us=1e6 / 60 - 4300, axes=("x",)):
    trs = [transport(a) for a in axes]
    return Sequence([("wait", 500)] + [("transport", tr) for tr in trs] + [("wait", gap_us / 2), ("pi",),
                    ("wait", gap_us / 2)] + [("transport", tr) for tr in trs] + [("wait", 500)], dt_us=5.0)


def test_filter_function_matches_monte_carlo():
    seq = Sequence([("wait", 8333.3), ("pi",), ("wait", 8333.3)], dt_us=10)
    g = seq.build()
    c = np.ones_like(g.t_us)
    spec = NoiseSpectrum(white=1e-2, flicker=0.5, f_min=0.5, lines=[(60, 0.3), (30, 0.2)])
    v = filter_variance(g, c, spec)
    tr = sample_noise(g, spec, np.random.default_rng(3), 800)
    ph = 2 * np.pi * np.sum(g.sign * c * tr, axis=1) * g.dt_us * 1e-6
    assert ph.var() == pytest.approx(v, rel=0.15)


def test_static_shift_cancels_in_symmetric_echo():
    g = symmetric_echo().build()
    c = coefficients(g, FlatPair(), lambda th: 1e-3 + 0 * th)
    assert abs(phase(g, c["static"], 1.0)) < 1e-9


def test_line_noise_cancels_for_whole_line_periods_between_transports():
    g = symmetric_echo().build()            # transports start 1/60 s apart
    c = coefficients(g, FlatPair(), lambda th: 0 * th)["field"]
    v_sync = filter_variance(g, c, NoiseSpectrum(lines=[(60.0, 1.0), (120.0, 1.0)]))
    g2 = symmetric_echo(gap_us=1e6 / 60 - 4300 + 4000).build()
    v_off = filter_variance(g2, coefficients(g2, FlatPair(), lambda th: 0 * th)["field"],
                            NoiseSpectrum(lines=[(60.0, 1.0), (120.0, 1.0)]))
    assert v_sync < 1e-5 * v_off  # limited by rounding 1/60 s to the 5 us grid


def test_transient_is_per_axis():
    g = symmetric_echo(axes=("x", "y")).build()
    dth, _ = LegTransient(d_theta_rad=1.0, tau_ms=1e9).offsets(g)
    legs = [dth[g.leg == k].max() for k in range(4)]
    assert legs == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_xy8_is_robust_and_cp_is_not():
    assert worst_input_contrast("XY-8", 9, eps_sigma=0.02, n_shots=60) > 0.995
    assert train_contrast("CP (pi_x)", 9, eps_sigma=0.02, n_shots=60) < 0.95
