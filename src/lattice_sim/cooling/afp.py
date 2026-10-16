"""Adiabatic-fast-passage microwave pulses on the coupled vibrational ladders.

Waveform (Wang 2016, Eq. 2.2): over a pulse of length tau the microwave
frequency and power follow

    f(t) = f0 + hsr * sign(t/tau - 1/2) * sqrt(1 - sin^4(pi t / tau))
    P(t) = P_max sin^4(pi t / tau)

so the Rabi frequency is Omega(t) = Omega0 sin^2(pi t / tau) (Omega0 from the
pulse power, ``rabi_from_power``) and
(f - f0)^2 / hsr^2 + Omega^2 / Omega0^2 = 1 (equal adiabaticity). ``hsr`` is
the lab's "half sweep range"; its sign sets the sweep direction (negative:
from f0 + |hsr| down to f0 - |hsr|).

The DDS updates the AFP parameters every 8 us (Wang p.24), so the pulse is a
sequence of piecewise-constant steps; in the frame that follows the
microwave phase the Hamiltonian is constant within a step and each step is
propagated exactly. Basis: |upper, n> (n = 0..N-1) then |lower, m>. Energies
are E/h in Hz measured from each state's own vibrational ground level;
``carrier_shift_hz`` is the carrier of this configuration relative to the
frequency reference. Coupling <lower m | upper n> * Omega / 2.
"""

import numpy as np
from scipy.integrate import solve_ivp

AFP_DURATION_S = 3e-3          # current sequence (lab, 2026-10-13)
AFP_UPDATE_S = 8e-6            # DDS update interval for AFP (Wang p.24)

# ---------------------------------------------------------------- power -> Rabi frequency
# The pulse power column of the DDS is in W and the field amplitude goes as
# sqrt(power) (Pow2AmpWord, dds-control firmware/NewDDS2-Cooling2.0-Cal-V2
# conversions.cpp). Calibration: "0.306 corresponds to 0.12 ms Blackman pulse
# for f4new transition (was 0.318, updated after replacing the power supply
# in early April 2022)" (dds-control scripts/Powerscan.py:317,
# scripts/Freqscan.py:337); f4new is the |4,-4> -> |3,-3> carrier. The
# firmware's Blackman window shapes the FIELD amplitude (power = powmax w^2,
# pulse.h ~945), so the pi pulse has area 0.42 T Omega_peak = 1/2 cycle:
# Omega_peak / 2pi = 9.92 kHz at 0.306 W.
CAL_POWER_W = 0.306
CAL_BLACKMAN_PI_S = 0.12e-3
BLACKMAN_AREA_FACTOR = 0.42    # mean of the Blackman amplitude window (pulse.h:945-947)
# ASSUMPTION: the calibration was taken on trapped atoms at theta = 0, where
# the ground-state carrier overlap is ~1; it is used as the bare (coupling-1)
# Rabi frequency. See data request 26.
CAL_IS_BARE_COUPLING = True
# Cooling-pulse powers (Freq-Scan-GUI output/Coolscan-manifest.json,
# 2026-10-13: "0.075/0.1425 W", p_m1 = 0.075, p_m2 = 0.1425; the same in
# both loops). Key: pulse kind (2 = the "-2" pulse, 1 = the "-1" pulse).
LAB_POWER_W = {2: 0.1425, 1: 0.075}


def rabi_from_power(power_w, cal_power_w=CAL_POWER_W, cal_pi_s=CAL_BLACKMAN_PI_S):
    """Peak bare Rabi frequency Omega/2pi (Hz) at a DDS power (W), from the
    Blackman pi-power calibration and Omega ~ sqrt(power)."""
    rabi_cal = 0.5 / (BLACKMAN_AREA_FACTOR * cal_pi_s)
    return rabi_cal * np.sqrt(np.asarray(power_w, float) / cal_power_w)


# Lab peak Rabi frequencies per pulse kind: 6.77 kHz (-2), 4.91 kHz (-1).
LAB_OMEGA_HZ = {k: float(rabi_from_power(p)) for k, p in LAB_POWER_W.items()}
# Note: the firmware loads each tic's frequency increment for a tic of
# L / (N - 1) but plays it for 8 us, so its sweep covers kappa =
# 8 us (N - 1) / L = 0.9973 of the range (3 ms) and is centred -(1 - kappa) HSR
# off the set frequency (+16 Hz at HSR -6 kHz); not modelled here.


def waveform(t, tau=AFP_DURATION_S, hsr_hz=-6e3):
    """(detuning of the microwave from f0 in Hz, Omega / Omega0) at times t."""
    s = np.sin(np.pi * np.asarray(t) / tau) ** 2
    df = hsr_hz * np.sign(np.asarray(t) / tau - 0.5) * np.sqrt(np.clip(1 - s**2, 0, None))
    return df, s


def ladder_hamiltonian(e_u, e_l, overlap, nu_hz, omega_hz, carrier_shift_hz=0.0):
    """H/h (Hz) in the frame rotating with a microwave at nu_hz (relative to
    the reference carrier). overlap[m, n] = <lower m | upper n>."""
    nu, nl = len(e_u), len(e_l)
    h = np.zeros((nu + nl, nu + nl))
    h[np.arange(nu), np.arange(nu)] = np.asarray(e_u) + carrier_shift_hz - nu_hz
    h[nu + np.arange(nl), nu + np.arange(nl)] = e_l
    h[nu:, :nu] = 0.5 * omega_hz * overlap
    h[:nu, nu:] = h[nu:, :nu].T
    return h


def afp_unitary(e_u, e_l, overlap, center_hz, hsr_hz, omega0_hz,
                carrier_shift_hz=0.0, tau=AFP_DURATION_S, dt=AFP_UPDATE_S):
    """Propagator of one AFP pulse centred at center_hz (relative to the
    reference carrier), piecewise-constant on the DDS update grid; omega0_hz
    is the peak bare Rabi frequency of the pulse (e.g. LAB_OMEGA_HZ[kind])."""
    n_steps = int(round(tau / dt))
    t_mid = (np.arange(n_steps) + 0.5) * dt
    df, s = waveform(t_mid, tau, hsr_hz)
    dim = len(e_u) + len(e_l)
    u = np.eye(dim, dtype=complex)
    for k in range(n_steps):
        h = ladder_hamiltonian(e_u, e_l, overlap, center_hz + df[k], omega0_hz * s[k], carrier_shift_hz)
        w, v = np.linalg.eigh(h)
        u = (v * np.exp(-2j * np.pi * w * dt)) @ (v.T @ u)
    return u


def transfer_probability(u):
    """|U|^2: element [j, i] is the probability to end in j starting in i."""
    return np.abs(u) ** 2


def two_level_transfer(overlap, detuning_hz, hsr_hz, omega0_hz,
                       tau=AFP_DURATION_S, dt=AFP_UPDATE_S):
    """Transfer of an isolated two-level line with coupling overlap * Omega0,
    whose resonance sits detuning_hz from the sweep centre."""
    u = afp_unitary([detuning_hz], [0.0], np.array([[overlap]]), 0.0, hsr_hz, omega0_hz,
                    tau=tau, dt=dt)
    return float(np.abs(u[1, 0]) ** 2)


def two_level_transfer_ode(overlap, detuning_hz, hsr_hz, omega0_hz,
                           tau=AFP_DURATION_S, rtol=1e-9):
    """Same as two_level_transfer with the smooth (continuous) waveform,
    integrated with an adaptive ODE solver in the lab frame of the sweep."""
    def rhs(t, y):
        df, s = waveform(t, tau, hsr_hz)
        h = np.array([[detuning_hz - df, 0.5 * omega0_hz * overlap * s],
                      [0.5 * omega0_hz * overlap * s, 0.0]])
        return -2j * np.pi * (h @ y)
    sol = solve_ivp(rhs, (0, tau), np.array([1.0 + 0j, 0j]), rtol=rtol, atol=1e-12,
                    max_step=tau / 2000)
    return float(np.abs(sol.y[1, -1]) ** 2)
