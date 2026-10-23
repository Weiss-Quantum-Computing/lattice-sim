"""Microwave pulses of the lab's DDS and the frequency scan they measure.

A frequency scan in the lab is a series of identical pulses whose set
(centre) frequency steps across the spectrum; the signal is the population
transferred to the lower hyperfine state. This module turns a list of
lines (frequency, coupling, initial population) into that signal for the
three pulse shapes the lab uses, with the time discretization, envelope and
sweep law of the DDS firmware.

Firmware conventions (dds-control, the microwave pulse compiler
``firmware/NewDDS2-Cooling2.0-Cal-V2``, which ``scripts/Freqscan.py:555``
and ``scripts/Powerscan.py`` call; ``-Cal`` has byte-identical PBlackman,
PAdiabatic and PConstant classes and the same conversions.cpp; ``-RF`` and
the run program ``NewDDS2-Cooling2.0`` carry an older pulse.h but do not
compile microwave pulses):

* Pulse file columns are [frequency (Hz), HSR (Hz; phase in units of pi for
  Blackman and square), length (ms), power (W), type] (``NewDDS.cpp:141-148``);
  type 1 = AFP (``PAdiabatic``), 2 = Blackman (``PBlackman``), 3 = square,
  i.e. constant amplitude (``PConstant``) (``NewDDS.cpp:158-230``).
* Timing. The DAQ plays one 32-bit register word per update-clock cycle,
  FIFOCLOCK = 1 MHz (``DDSconstants.h:17``); ``Time2Tics(s) = floor(s *
  FIFOCLOCK / 8)`` (``conversions.cpp:289-292``). Blackman: 1 + Time2Tics(4 L)
  tics of 2 words (``pulse.h:888, 958-977``), i.e. M = floor(L / 2 us)
  amplitude steps of 2 us. AFP: Time2Tics(L) tics of 8 words
  (``pulse.h:1041, 1132-1151``), i.e. N = floor(L / 8 us) tics of 8 us.
  Square: 1 + Time2Tics(8 L) tics of 1 word (``pulse.h:305, 375-379``), a
  constant amplitude for floor(L / 1 us) us. The Freq-Scan-GUI states the
  same grid (``ddsgui/config.py:68-80``: AFP 8 us, Blackman 2 us, square
  1 us). Each pulse's first tic is 16 register writes (16 us) with the
  amplitude still zero for Blackman and AFP (the register object starts
  zeroed, ``ad9852data.h:383-400``); it is a delay, not modelled. It does
  not change the transfer, but the real drive (and the AFP sweep centre)
  starts 16 us after the pulse does, while the time axis here starts with
  the drive: add 16 us when comparing with time-resolved data or external
  timing.
* Power -> field. The power column is in W; ``Pow2AmpWord``
  (``conversions.cpp:112-282``) maps it through a measured 2015 table
  (0 .. POWMAX = 5.188 W, ``DDSconstants.h``) to the 12-bit DDS amplitude
  word, so the delivered power follows the requested power and the field
  amplitude, hence the Rabi frequency, goes as sqrt(power). The lab's own
  numbers agree: a 5x longer Blackman pi pulse uses 1/25 of the power
  (``Freqscan.py:336-337``) and pi/2 uses P/4 (``Freqscan.py:361``,
  ``Powerscan.py:320``); ``PIPULSEAREA`` is "time*sqrt(power)"
  (``DDSconstants.h``). Assumed exact here (POWER_CALIBRATION_EXACT); the
  12-bit amplitude quantization (1/4095 of the 5.188 W amplitude) is not
  modelled.
* Blackman. Power = powmax * w(k/M)^2 with w(x) = 0.42 - 0.5 cos(2 pi x) +
  0.08 cos(4 pi x), "power is amplitude squared" (``pulse.h:945-947``), for
  k = 1 .. M (``pulse.h:939``): the window shapes the FIELD amplitude,
  constant frequency. The pulse area is 0.42 M (2 us) times the peak Rabi
  frequency, exactly (the cosine sums vanish over k = 1 .. M). The
  commented-out PPi/PPiHalf classes (``pulse.h:1216-1238``, inside /* */)
  use L = (0.5 / rabi) / 0.30265 "from integrating power of Blackman
  pulse"; 0.30265 is close to the power integral of the window (0.3046),
  which would be the area only if the Rabi frequency went as the power. They
  are not compiled; this module uses the amplitude area 0.42.
* AFP. Power = powmax * sin^4(pi (i - 1) / (N - 1)) (``pulse.h:1119``), so
  the field amplitude is sin^2, held for each 8 us tic i = 1 .. N - 1. The
  frequency starts at f - HSR (``pulse.h:1063``) in the DDS chirp mode
  (``pulse.h:1049-1052``); each tic loads the frequency increment
  Freq1(tau_i) - Freq1(tau_{i-1}), tau_i = i / (N - 1) (``pulse.h:1124-1125``),
  spread over the tic as a per-ramp-clock step (RAMPRATE = 1 MHz,
  ``pulse.h:1127``), with Freq1(f0, HSR, tau) = f0 + sgn(tau - 1/2) HSR
  sqrt(1 - sin^4(pi tau)) (``conversions.cpp:309-318``). The drive therefore
  sweeps from f - HSR to f + HSR through f at mid-pulse: HSR is the HALF
  sweep range ("HSR Half frequency Sweeping Range", ``pulse.h:1040``), signed
  (the lab uses HSR = -4 kHz, a downward sweep, ``Freqscan.py:325``).
  Amplitude and detuning satisfy (Omega / Omega_peak)^2 + (Delta / HSR)^2 = 1:
  for Omega_peak = |HSR| the effective field is constant. The increment is
  computed for a tic of L / (N - 1) but plays for 8 us, so the sweep covers
  a fraction kappa = 8 us (N - 1) / L of the intended one, centred
  -(1 - kappa) HSR from f (+11 Hz for 3 ms, HSR = -4 kHz); modelled. The
  commented sech^2 / sech^4 envelopes and Freq2 (sech law, 4.852) are not
  used (``pulse.h:1116-1118, 1122-1123``).
* Square. PConstant holds a constant amplitude sqrt(power) at the set
  frequency for floor(L / 1 us) us.

Calibration. The lab sets power in W and calibrates it per transition by a
power scan: "0.306 corresponds to 0.12 ms Blackman pulse for f4new
transition" (|4,-4> -> |3,-3>; ``Powerscan.py:318``, ``Freqscan.py:337``,
after the April 2022 power-supply change). With the amplitude area this
gives a peak Rabi frequency Omega/2pi = 1 / (2 x 0.42 x 0.12 ms) = 9.92 kHz
at 0.306 W, and Omega/2pi = 9.92 kHz sqrt(P / 0.306 W) at any power. The
calibration was taken on trapped atoms, i.e. on the lattice carrier, whose
coupling s (below) is < 1; here it is assigned to s = 1 (CAL_IS_BARE_ATOM).
Typical lab pulses: AFP 3 ms, 0.08 W, HSR -4 kHz for carrier and sideband
scans (``Freqscan.py:322-325``; ``ddsgui/daylog.py:11``); Blackman
0.12-0.6 ms at the pi power for the transition ladder (``Freqscan.py:334-392``);
the per-transition recipe is a low-power Blackman frequency scan, a power
scan for the pi power, and P/4 for pi/2. Other transitions scale the Rabi
frequency by sqrt(hop_weight / 28) (``hop_weight``).

Model. Each line |upper, n> -> |lower, n'> (3D vibrational states) is a
two-level system driven with Rabi frequency Omega(t) = Omega_peak env(t) s,
where s = |<n'|n>| is the product over the three axes of the 1D overlaps
(carrier: the diagonal overlaps; a sideband on axis a: the off-diagonal
overlap on a and the diagonals on the other two; the degenerate final
states on the own and the neighbouring sites summed in quadrature,
``lines_from_spectrum``), and detuning
Delta(t) = f_set + sweep(t) - f_line. The lines that start in one initial
state n combine as successive independent transfers, so the scan signal is
sum_n p(n) [1 - prod_lines (1 - P(s, f_set - f_line))] over the thermal
populations p(n) (``combine_lines``): the plain sum for resolved lines, at
most p(n) where one pulse reaches several lines of n (an AFP sweeping over
the sidebands of two pairs). Approximations:

* lines independent: valid when the Rabi frequency is small compared with
  the spacing to the next line (the trap frequencies, 12.5-19 kHz).
  ``coupled_transfer`` drives all levels of one axis together; against it
  (two displaced 19 kHz wells, 1 uK, ``tests/test_microwave.py``) the
  independent lines are within 0.015 for a 0.6 ms Blackman pi pulse and
  0.01 for a 0.6 ms square pi pulse, off by > 0.04 for the 0.12 ms
  Blackman pi pulse (peak Rabi 9.9 kHz), and by 0.01-0.06 for the 3 ms,
  0.08 W AFP near the sideband, where the off-resonant carrier light-shifts
  the sideband (~Omega^2 / 4 f_trap ~ 0.3 kHz);
* rotating-wave approximation (the Rabi frequency is ~1e-6 of the 9.2 GHz
  carrier);
* no decoherence during the pulse; optionally a quasi-static detuning
  spread (shot-to-shot field or light-shift noise, Gaussian, rms
  ``spread_hz``), which averages the response over the detuning;
* the final state is not resolved: every |lower, n'> counts as transferred.

Numerics. For one pulse, P depends only on (s, detuning), so a response
table on a grid in s (P / s^2 is stored, exact at small s) and in detuning
is built once per pulse by vectorized piecewise-constant 2x2 propagation
over the firmware steps (AFP tics subdivided, frequency at each substep's
midpoint); ``transfer_reference`` (scipy.integrate.solve_ivp of the
Schroedinger equation with the frequency ramped linearly inside each tic)
is the independent check. Beyond ``response_reach_hz`` (where P < 1e-7 at
the largest s) a Blackman or AFP pulse is taken to transfer nothing. A scan
multiplies each line's interpolated response into its initial state's
survival on the table's detuning grid.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

# ---------------------------------------------------------------- firmware
UPDATE_CLOCK_HZ = 1.0e6              # DDSconstants.h FIFOCLOCK: one register word per cycle
SQUARE_STEP_S = 1.0e-6               # PConstant: 1 word per tic
BLACKMAN_STEP_S = 2.0e-6             # PBlackman: 2 words (amplitude) per tic
AFP_STEP_S = 8.0e-6                  # PAdiabatic: 8 words (amplitude + delta frequency) per tic
RAMP_CLOCK_HZ = 1.0e6                # DDSconstants.h RAMPRATE = EXTCLOCK * 0.1
BLACKMAN_COEFFS = (0.42, 0.5, 0.08)  # pulse.h:945-947
POWMAX_W = 5.188                     # DDSconstants.h POWMAX, top of the Pow2AmpWord table
POWER_CALIBRATION_EXACT = True       # delivered power = requested power (Pow2AmpWord table)
FLOOR_EPS = 1e-9                     # guards floor() against 59.999999 from the float product
SHAPES = ("square", "blackman", "afp")

# ------------------------------------------------------------- calibration
CAL_BLACKMAN_LENGTH_S = 0.12e-3      # Powerscan.py:317-318
CAL_BLACKMAN_PI_POWER_W = 0.306      # "0.306 corresponds to 0.12 ms Blackman pulse for f4new"
CAL_IS_BARE_ATOM = True              # the calibration is assigned to coupling s = 1

# --------------------------------------------------------------- lab pulses
LAB_PULSES = {                       # (length s, power W, HSR Hz)
    "afp": (3.0e-3, 0.08, -4.0e3),   # Freqscan.py:322-325, ddsgui/daylog.py:11
    "blackman": (0.6e-3, None, 0.0),  # Freqscan.py:336 (0.12*5 ms); power None = the pi power
    "square": (0.6e-3, None, 0.0),
}

# ----------------------------------------------------------------- numerics
AFP_SUBSTEPS = 2                     # propagation substeps per 8 us AFP tic
DETUNING_STEPS_PER_INVERSE_DURATION = {"square": 24, "blackman": 12, "afp": 6}   # table step 1 / (k T)
PROBE_STEPS_PER_INVERSE_DURATION = 4  # detuning step of the reach probe
SUPPORT_THRESHOLD = 1e-7             # P below this (at the largest s) counts as no response
SUPPORT_MARGIN_INVERSE_DURATIONS = 4.0
SCALE_POINTS_PER_AREA = {"square": 16, "blackman": 16, "afp": 8}   # s points per cycle of area at s = 1
TABLE_DTYPE = np.float32             # response tables (direct evaluation stays float64)
MIN_SCALE_POINTS = 41
MAX_SCALE_POINTS = 161
SMALL_SCALE = 1e-4                   # s at which the s -> 0 limit of P / s^2 is taken
SPREAD_NODES = 24                    # Gauss-Hermite nodes for a detuning spread (direct evaluation)
SPREAD_SIGMAS = 4.0                  # Gaussian kernel half-width (table convolution)
MAX_DN = 2                           # sidebands kept: |dn| <= 2 on one axis
MIN_POPULATION = 1e-5                # initial 3D states below this population are dropped


# ===================================================================== pulses
@dataclass(frozen=True)
class Pulse:
    """One DDS pulse as the lab writes it, in SI units.

    ``rabi_hz`` is the peak Rabi frequency Omega/2pi of a line with coupling
    s = 1 at the set power (``rabi_from_power``); ``hsr_hz`` the AFP half
    sweep range, signed as in the pulse file (ignored for the other shapes).
    """
    shape: str
    duration_s: float
    rabi_hz: float
    hsr_hz: float = 0.0

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, not {self.shape!r}")
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")

    def with_rabi(self, rabi_hz):
        return Pulse(self.shape, self.duration_s, float(rabi_hz), self.hsr_hz)


def blackman_window(x):
    a0, a1, a2 = BLACKMAN_COEFFS
    x = np.asarray(x, float)
    return a0 - a1 * np.cos(2 * np.pi * x) + a2 * np.cos(4 * np.pi * x)


def _count(length_s, step_s):
    return int(np.floor(length_s / step_s + FLOOR_EPS))


def _afp_frequency(tau, hsr_hz):
    """Freq1 - f0 (conversions.cpp:309-318)."""
    return np.sign(tau - 0.5) * hsr_hz * np.sqrt(np.clip(1.0 - np.sin(np.pi * tau) ** 4, 0.0, None))


def pulse_segments(pulse: Pulse):
    """Firmware time steps: (dt_s, amplitude, f_start_hz, f_end_hz) arrays.

    amplitude is the field amplitude relative to its peak (Rabi frequency =
    pulse.rabi_hz x amplitude x s); f_start/f_end the drive frequency minus
    the set frequency at the start and end of the step (linear in between).
    """
    if pulse.shape == "square":
        n = _count(pulse.duration_s, SQUARE_STEP_S)
        if n < 1:
            raise ValueError("square pulse shorter than one 1 us step")
        z = np.zeros(1)
        return np.array([n * SQUARE_STEP_S]), np.ones(1), z, z.copy()
    if pulse.shape == "blackman":
        m = _count(pulse.duration_s, BLACKMAN_STEP_S)
        if m < 2:
            raise ValueError("Blackman pulse shorter than two 2 us steps")
        k = np.arange(1, m + 1)
        amp = np.clip(blackman_window(k / m), 0.0, None)
        z = np.zeros(m)
        return np.full(m, BLACKMAN_STEP_S), amp, z, z.copy()
    n = _count(pulse.duration_s, AFP_STEP_S)
    if n < 3:
        raise ValueError("AFP pulse shorter than three 8 us tics")
    i = np.arange(1, n)
    amp = np.sin(np.pi * (i - 1) / (n - 1)) ** 2
    kappa = AFP_STEP_S * (n - 1) / pulse.duration_s
    hsr = pulse.hsr_hz
    g = -hsr + kappa * (_afp_frequency(np.arange(n) / (n - 1), hsr) + hsr)
    return np.full(n - 1, AFP_STEP_S), amp, g[:-1], g[1:]


def _steps(pulse: Pulse, substeps=None):
    """Propagation steps (dt, amplitude, frequency offset at the step midpoint)."""
    substeps = AFP_SUBSTEPS if substeps is None else substeps
    dt, amp, f0, f1 = pulse_segments(pulse)
    if pulse.shape != "afp" or substeps <= 1:
        return dt, amp, 0.5 * (f0 + f1)
    frac = (np.arange(substeps) + 0.5) / substeps
    off = (f0[:, None] + frac[None, :] * (f1 - f0)[:, None]).ravel()
    return np.repeat(dt / substeps, substeps), np.repeat(amp, substeps), off


def pulse_area(pulse: Pulse):
    """Integral of the Rabi frequency for s = 1, in cycles (P = sin^2(pi area)
    on resonance for a pulse without sweep)."""
    dt, amp, _, _ = pulse_segments(pulse)
    return float(pulse.rabi_hz * np.sum(amp * dt))


def envelope_area_s(shape, duration_s):
    """Integral of the relative amplitude over the pulse (s): 0.42 M x 2 us for Blackman."""
    return float(np.sum(np.prod(pulse_segments(Pulse(shape, duration_s, 1.0))[:2], axis=0)))


def rabi_from_power(power_w, cal_power_w=CAL_BLACKMAN_PI_POWER_W, cal_length_s=CAL_BLACKMAN_LENGTH_S):
    """Peak Rabi frequency (Hz, s = 1) at a power, from the lab's Blackman pi-power calibration."""
    rabi_cal = 0.5 / envelope_area_s("blackman", cal_length_s)
    return rabi_cal * np.sqrt(np.asarray(power_w, float) / cal_power_w)


def pi_power(shape, duration_s, cal_power_w=CAL_BLACKMAN_PI_POWER_W, cal_length_s=CAL_BLACKMAN_LENGTH_S,
             scale=1.0):
    """Power (W) of a resonant pi pulse of a square or Blackman shape on a line of coupling s."""
    if shape == "afp":
        raise ValueError("an AFP has no pi power")
    rabi = 0.5 / (envelope_area_s(shape, duration_s) * scale)
    return float(cal_power_w * (rabi / rabi_from_power(cal_power_w, cal_power_w, cal_length_s)) ** 2)


def hop_weight(m_upper, m_lower):
    """Clebsch-Gordan weight of |4,m_upper> -> |3,m_lower> in units of 1/32
    (Freq-Scan-GUI ddsgui/paths.py:20-45, 108-122: 2|<S_q>|^2, spherical S_q,
    equal sigma+/pi/sigma- field amplitudes). The stretched hop is 28; the
    Rabi frequency of another hop is sqrt(weight / 28) times the stretched one."""
    if m_lower == m_upper:
        return 16 - m_upper ** 2
    if m_lower == m_upper + 1:
        return (4 - m_upper) * (3 - m_upper) // 2
    if m_lower == m_upper - 1:
        return (4 + m_upper) * (3 + m_upper) // 2
    raise ValueError(f"|4,{m_upper:+d}> -> |3,{m_lower:+d}> is not a microwave hop")


# ================================================================ two-level
def transfer(pulse: Pulse, scale, detuning_hz, spread_hz=0.0, chunk=200_000):
    """Transferred population of one line: coupling s, detuning (set frequency
    minus line frequency, Hz); arrays broadcast. Piecewise-constant 2x2
    propagation over the firmware steps. With spread_hz > 0 the detuning is
    averaged over a Gaussian of that rms (Gauss-Hermite)."""
    s, d = np.broadcast_arrays(np.asarray(scale, float), np.asarray(detuning_hz, float))
    if spread_hz > 0:
        x, w = np.polynomial.hermite.hermgauss(SPREAD_NODES)
        out = np.zeros(s.shape)
        for xi, wi in zip(x, w):
            out += wi / np.sqrt(np.pi) * transfer(pulse, s, d + np.sqrt(2) * spread_hz * xi, 0.0, chunk)
        return out
    shape = s.shape
    s, d = s.ravel(), d.ravel()
    out = np.empty(s.size)
    for lo in range(0, s.size, chunk):
        out[lo:lo + chunk] = _propagate(pulse, s[lo:lo + chunk][None, :], d[lo:lo + chunk][None, :])[0]
    return out.reshape(shape)


def _propagate(pulse, s, d, progress=None, cancelled=None, dtype=np.float64, both=False):
    """|c_lower|^2 after the pulse for broadcastable 2D arrays s, d (start in
    upper); with both=True (|c_upper|^2, |c_lower|^2).

    H/h = (Omega/2) sigma_x - (Delta/2) sigma_z, U = cos(th) - i sin(th) n.sigma
    with th = pi dt W, W = sqrt(Omega^2 + Delta^2), real arithmetic."""
    dt, amp, off = _steps(pulse)
    s = np.asarray(s, dtype)
    d = np.asarray(d, dtype)
    shape = np.broadcast_shapes(s.shape, d.shape)
    ar, ai = np.ones(shape, dtype), np.zeros(shape, dtype)
    br, bi = np.zeros(shape, dtype), np.zeros(shape, dtype)
    s2 = s * s
    tiny = dtype(1e-30)
    n = dt.size
    report = max(1, n // 50)
    for k in range(n):
        if cancelled is not None and k % report == 0:
            if cancelled():
                return None
            if progress is not None:
                progress(k, n)
        om = dtype(pulse.rabi_hz * amp[k])
        dk = d + dtype(off[k])
        w = np.sqrt(s2 * (om * om) + dk * dk)
        th = w * dtype(np.pi * dt[k])
        c = np.cos(th)
        sinc = np.sin(th) / np.maximum(w, tiny)               # sin(th) / W
        sx = sinc * (s * om)                                  # sin(th) n_x
        sz = sinc * (-dk)                                     # sin(th) n_z
        ar, ai, br, bi = (c * ar + sz * ai + sx * bi, c * ai - sz * ar - sx * br,
                          c * br + sx * ai - sz * bi, c * bi - sx * ar + sz * br)
    if both:
        return (ar * ar + ai * ai).astype(np.float64), (br * br + bi * bi).astype(np.float64)
    return (br * br + bi * bi).astype(np.float64)


def transfer_square_analytic(pulse: Pulse, scale, detuning_hz):
    """Rabi formula for the square pulse: (s Omega / W)^2 sin^2(pi W T)."""
    t = pulse_segments(pulse)[0][0]
    om = pulse.rabi_hz * np.asarray(scale, float)
    w2 = om ** 2 + np.asarray(detuning_hz, float) ** 2
    w = np.sqrt(w2)
    return np.where(w2 > 0, om ** 2 / np.where(w2 > 0, w2, 1.0), 0.0) * np.sin(np.pi * w * t) ** 2


def transfer_reference(pulse: Pulse, scale, detuning_hz, rtol=1e-10, atol=1e-12):
    """Independent reference: solve_ivp (DOP853) of the Schroedinger equation
    segment by segment, the AFP frequency ramped linearly inside each tic."""
    from scipy.integrate import solve_ivp
    dt, amp, f0, f1 = pulse_segments(pulse)
    y = np.array([1.0, 0.0, 0.0, 0.0])          # Re a, Im a, Re b, Im b
    s = float(scale)
    for k in range(dt.size):
        om = 2 * np.pi * pulse.rabi_hz * amp[k] * s
        slope = (f1[k] - f0[k]) / dt[k]

        def rhs(t, y, om=om, f0k=f0[k], slope=slope):
            delta = 2 * np.pi * (detuning_hz + f0k + slope * t)
            a = y[0] + 1j * y[1]
            b = y[2] + 1j * y[3]
            da = -1j * (-0.5 * delta * a + 0.5 * om * b)
            db = -1j * (0.5 * om * a + 0.5 * delta * b)
            return [da.real, da.imag, db.real, db.imag]
        y = solve_ivp(rhs, (0.0, dt[k]), y, method="DOP853", rtol=rtol, atol=atol).y[:, -1]
    return float(y[2] ** 2 + y[3] ** 2)


# ============================================================ response table
def response_reach_hz(pulse: Pulse, detuning_max_hz, scale_max=1.0):
    """Largest |detuning| (<= detuning_max_hz) at which the pulse still transfers
    more than SUPPORT_THRESHOLD of a line of coupling scale_max, plus a margin of
    SUPPORT_MARGIN_INVERSE_DURATIONS / T. Square pulses (power-law wings): detuning_max_hz.
    Found by propagating one row on a coarse detuning grid."""
    if pulse.shape == "square":
        return float(detuning_max_hz)
    step = 1.0 / (PROBE_STEPS_PER_INVERSE_DURATION * pulse.duration_s)
    n = int(np.ceil(detuning_max_hz / step))
    d = step * np.arange(-n, n + 1)
    p = _propagate(pulse, np.array([[scale_max]]), d[None, :], dtype=TABLE_DTYPE)[0]
    big = np.nonzero(p > SUPPORT_THRESHOLD)[0]
    if big.size == 0:
        return min(float(detuning_max_hz), SUPPORT_MARGIN_INVERSE_DURATIONS / pulse.duration_s)
    reach = float(np.max(np.abs(d[big]))) + step + SUPPORT_MARGIN_INVERSE_DURATIONS / pulse.duration_s
    return min(float(detuning_max_hz), reach)


@dataclass
class Response:
    """P(s, detuning) of one pulse on a grid; P / s^2 is stored (``q``)."""
    pulse: Pulse
    scale: np.ndarray                  # uniform s grid from 0
    detuning_hz: np.ndarray            # uniform, symmetric about 0
    q: np.ndarray                      # (n_scale, n_detuning) = P / s^2
    spread_hz: float = 0.0
    seconds: float = 0.0
    raw_q: np.ndarray = field(default=None, repr=False)   # before the detuning spread

    @property
    def step_hz(self):
        return float(self.detuning_hz[1] - self.detuning_hz[0])

    def probability(self):
        return self.q * self.scale[:, None] ** 2

    def _scale_index(self, s):
        u = np.asarray(s, float) / (self.scale[1] - self.scale[0])
        j = np.clip(np.floor(u).astype(int), 0, self.scale.size - 2)
        return j, u - j

    def at(self, scale, detuning_hz):
        """Bilinear interpolation; 0 outside the detuning range, linear extrapolation in s."""
        s, d = np.broadcast_arrays(np.asarray(scale, float), np.asarray(detuning_hz, float))
        j, t = self._scale_index(s)
        v = (d - self.detuning_hz[0]) / self.step_hz
        inside = (v >= 0) & (v <= self.detuning_hz.size - 1)
        k = np.clip(np.floor(v).astype(int), 0, self.detuning_hz.size - 2)
        r = np.clip(v - k, 0.0, 1.0)
        q = ((1 - t) * ((1 - r) * self.q[j, k] + r * self.q[j, k + 1])
             + t * ((1 - r) * self.q[j + 1, k] + r * self.q[j + 1, k + 1]))
        return np.where(inside, np.clip(q * s * s, 0.0, 1.0), 0.0)

    def row(self, scale):
        """P(detuning) on the table grid for one coupling."""
        j, t = self._scale_index(scale)
        return np.clip(((1 - t) * self.q[j] + t * self.q[j + 1]) * float(scale) ** 2, 0.0, 1.0)

    def with_spread(self, spread_hz):
        """The same pulse averaged over a Gaussian detuning spread (rms, Hz)."""
        base = self.raw_q if self.raw_q is not None else self.q
        q = smear(base, self.step_hz, spread_hz)
        return Response(self.pulse, self.scale, self.detuning_hz, q, float(spread_hz), self.seconds, base)

    def scan(self, lines, f_hz):
        """Transferred population at set frequencies f_hz: sum over initial states
        of p(n) [1 - prod_lines (1 - P_line)] (``combine_lines``).

        Each line's response (its s row, shifted to the line's frequency with
        linear weights) is multiplied into its state's survival on a grid of
        the table's detuning step, which is then interpolated onto f_hz."""
        f = np.asarray(f_hz, float)
        h = self.step_hz
        n_d = self.detuning_hz.size
        lo = float(f.min()) - (n_d + 1) * h
        n_grid = int(np.ceil((float(f.max()) - float(f.min())) / h)) + 2 * n_d + 4
        pos = (lines.detuning_hz + self.detuning_hz[0] - lo) / h
        near = (pos >= 0) & (pos < n_grid - n_d - 2) & (lines.population > 0)
        sel = lines.select(near)
        if len(sel) == 0:
            return np.zeros_like(f)
        pos = pos[near]
        k0 = np.floor(pos).astype(int)
        r = (pos - k0)[:, None]
        j, t = self._scale_index(sel.scale)
        rows = ((1 - t)[:, None] * self.q[j] + t[:, None] * self.q[j + 1]) * (sel.scale ** 2)[:, None]
        pad = np.zeros((rows.shape[0], n_d + 2))
        pad[:, 1:-1] = rows
        keep_line = 1.0 - np.clip(r * pad[:, :-1] + (1 - r) * pad[:, 1:], 0.0, 1.0)     # (lines, n_d + 1)
        survive = np.ones((sel.n_states, n_grid))
        for i in range(len(sel)):
            survive[sel.state[i], k0[i]:k0[i] + n_d + 1] *= keep_line[i]
        total = sel.state_population @ (1.0 - survive)
        return np.interp(f, lo + h * np.arange(n_grid), total)

    def scan_sum(self, lines, f_hz):
        """sum_lines p P(s, f - f_line), the plain sum (right for resolved lines; can
        exceed 1 where one pulse drives several lines of an initial state). The
        lines are deposited on the detuning grid and convolved with each s row (FFT)."""
        f = np.asarray(f_hz, float)
        h = self.step_hz
        n_half = (self.detuning_hz.size - 1) // 2
        lo = float(f.min()) - n_half * h
        n_grid = int(np.ceil((float(f.max()) + n_half * h - lo) / h)) + 2
        pos = (lines.detuning_hz - lo) / h
        keep = (pos >= 0) & (pos < n_grid - 1) & (lines.population > 0)
        pos, s, p = pos[keep], lines.scale[keep], lines.population[keep]
        k = np.floor(pos).astype(int)
        r = pos - k
        j, t = self._scale_index(s)
        c = p * s * s
        sticks = np.zeros((self.scale.size, n_grid))
        for jj, wj in ((j, 1 - t), (j + 1, t)):
            np.add.at(sticks, (jj, k), c * wj * (1 - r))
            np.add.at(sticks, (jj, k + 1), c * wj * r)
        used = np.nonzero(sticks.any(axis=1))[0]
        if used.size == 0:
            return np.zeros_like(f)
        conv = fftconvolve(sticks[used], self.q[used], mode="full", axes=1).sum(axis=0)
        grid = lo + h * np.arange(n_grid)
        return np.interp(f, grid, conv[n_half:n_half + n_grid])

    def transfer_at(self, lines, f_set_hz):
        """Transferred population at one set frequency (``combine_lines``)."""
        return float(combine_lines(lines, self.at(lines.scale, f_set_hz - lines.detuning_hz)))


def smear(q, step_hz, spread_hz):
    """Convolve the rows of q with a normalized Gaussian of rms spread_hz (0 outside the grid)."""
    if spread_hz <= 0 or spread_hz < 0.25 * step_hz:
        return q
    half = int(np.ceil(SPREAD_SIGMAS * spread_hz / step_hz))
    x = np.arange(-half, half + 1) * step_hz
    kern = np.exp(-0.5 * (x / spread_hz) ** 2)
    kern /= kern.sum()
    return fftconvolve(q, kern[None, :], mode="same", axes=1)


def table_grid(pulse: Pulse, reach_hz, scale_max=1.0):
    """(scale grid, detuning grid) of the response table of a pulse covering |detuning| <= reach_hz."""
    step = 1.0 / (DETUNING_STEPS_PER_INVERSE_DURATION[pulse.shape] * pulse.duration_s)
    n_half = max(8, int(np.ceil(reach_hz / step)))
    detuning = step * np.arange(-n_half, n_half + 1)
    area = abs(pulse_area(pulse)) * scale_max
    n_scale = int(np.clip(np.ceil(SCALE_POINTS_PER_AREA[pulse.shape] * area) + 1, MIN_SCALE_POINTS,
                          MAX_SCALE_POINTS))
    return np.linspace(0.0, scale_max, n_scale), detuning


def build_response(pulse: Pulse, detuning_max_hz, spread_hz=0.0, scale_max=1.0, progress=None,
                   cancelled=None):
    """Response table of a pulse for |detuning| <= detuning_max_hz (Blackman and
    AFP: at most ``response_reach_hz``; zero beyond). Returns None if cancelled."""
    import time
    t0 = time.perf_counter()
    reach = response_reach_hz(pulse, detuning_max_hz, scale_max)
    scale, detuning = table_grid(pulse, reach + SPREAD_SIGMAS * spread_hz, scale_max)
    s_eval = scale.copy()
    s_eval[0] = SMALL_SCALE
    n_half = (detuning.size - 1) // 2
    if pulse.shape == "square":
        p = transfer_square_analytic(pulse, s_eval[:, None], detuning[None, :])
    else:
        symmetric = pulse.shape == "blackman"       # real symmetric envelope, no sweep: P(-d) = P(d)
        d_eval = detuning[n_half:] if symmetric else detuning
        p = _propagate(pulse, s_eval[:, None], d_eval[None, :], progress=progress,
                       cancelled=cancelled if cancelled is not None else (lambda: False), dtype=TABLE_DTYPE)
        if p is None:
            return None
        if symmetric:
            p = np.concatenate([p[:, :0:-1], p], axis=1)
    q = p / (s_eval[:, None] ** 2)
    out = Response(pulse, scale, detuning, q, 0.0, 0.0, q)
    if spread_hz > 0:
        out = out.with_spread(spread_hz)
    out.seconds = time.perf_counter() - t0
    return out


# ===================================================================== lines
@dataclass
class LineList:
    """Lines of a spectrum: detuning from the free-atom line (Hz), coupling
    s = |<n'|n>|, population of the initial state, the initial state's index
    (lines sorted by it), and labels (axis: '' for the carrier; dn: quanta
    added on that axis). ``kept_population``: total population of the
    initial states represented."""
    detuning_hz: np.ndarray
    scale: np.ndarray
    population: np.ndarray
    state: np.ndarray
    axis: np.ndarray
    dn: np.ndarray
    kept_population: float = 1.0
    harmonic: bool = False             # energies from harmonic ladders (no per-axis levels given)

    def __len__(self):
        return int(self.detuning_hz.size)

    @property
    def n_states(self):
        return int(self.state.max()) + 1 if self.state.size else 0

    @property
    def state_population(self):
        pop = np.zeros(self.n_states)
        pop[self.state] = self.population
        return pop

    def slice(self, lo, hi):
        """Lines lo .. hi - 1, state indices kept."""
        s = slice(lo, hi)
        return LineList(self.detuning_hz[s], self.scale[s], self.population[s], self.state[s], self.axis[s],
                        self.dn[s], self.kept_population, self.harmonic)

    def select(self, mask):
        """Subset of the lines; state indices renumbered 0 .. n - 1 in order."""
        state = self.state[mask]
        _, renum = np.unique(state, return_inverse=True)
        return LineList(self.detuning_hz[mask], self.scale[mask], self.population[mask], renum.ravel(),
                        self.axis[mask], self.dn[mask], self.kept_population, self.harmonic)


def make_lines(detuning_hz, scale, population, state=None, axis=None, dn=None, kept_population=None):
    """A LineList from arrays, sorted by initial state (state=None: each line its own state)."""
    detuning_hz = np.atleast_1d(np.asarray(detuning_hz, float))
    n = detuning_hz.size
    scale = np.broadcast_to(np.asarray(scale, float), (n,)).copy()
    population = np.broadcast_to(np.asarray(population, float), (n,)).copy()
    state = np.arange(n) if state is None else np.broadcast_to(np.asarray(state, int), (n,))
    axis = np.full(n, "") if axis is None else np.broadcast_to(np.asarray(axis), (n,))
    dn = np.zeros(n, int) if dn is None else np.broadcast_to(np.asarray(dn, int), (n,))
    order = np.argsort(state, kind="stable")
    _, renum = np.unique(state[order], return_inverse=True)
    if kept_population is None:
        _, first = np.unique(renum, return_index=True)
        kept_population = float(population[order][first].sum())
    return LineList(detuning_hz[order], scale[order], population[order], renum.ravel(), axis[order].copy(),
                    dn[order].copy(), float(kept_population))


def combine_lines(lines: LineList, p_lines):
    """Transferred population from per-line probabilities (last axis: lines first).

    Lines that start in the same initial state combine as successive
    independent transfers, p(n) [1 - prod (1 - P_line)]: the plain sum for
    resolved lines (one P_line non-zero at a time), and at most p(n) where one
    pulse drives several lines of the same state (e.g. an AFP sweeping over
    sidebands of two axes). Their coherent multi-level dynamics is not
    modelled (``coupled_transfer`` is the reference)."""
    p = np.clip(np.asarray(p_lines, float), 0.0, 1.0 - 1e-12)
    log_keep = np.zeros((lines.n_states,) + p.shape[1:])
    np.add.at(log_keep, lines.state, np.log1p(-p))
    return np.tensordot(lines.state_population, 1.0 - np.exp(log_keep), axes=(0, 0))


def _franck_condon_all_sites(axis_data):
    """|<n'|n>|^2 of one axis summed over the final sites k
    (``franck_condon_sites``); the k = 0 matrix for a spectrum without them."""
    sites = axis_data.get("franck_condon_sites")
    if not sites:
        return np.asarray(axis_data["franck_condon"], float)
    return sum(np.asarray(f, float) for f in sites.values())


def lines_from_spectrum(sp, states=None, max_dn=MAX_DN, min_population=MIN_POPULATION, max_states=None):
    """Lines of ``sidebands.spectrum`` output for any (upper, lower) pair.

    Initial 3D states are products of the per-axis thermal weights (those
    below min_population dropped, and all but the max_states most populated;
    ``kept_population`` is what is left); from each, the carrier and the
    |dn| <= max_dn sidebands on one axis at a time (sidebands on two axes at
    once, second order in the off-diagonal overlaps, are left out;
    ``strength_outside_lines`` says how much that and the level cut leave
    out). Line frequency = sum over axes of E(upper, n_a) - E(lower, n'_a),
    measured from the free-atom line, as in ``spectrum``. ``states``
    defaults to the order of each axis' levels dict. A spectrum without the
    per-axis ``levels`` (the lattice-grid view drops them) gets harmonic
    ladders at f_upper_hz / f_lower_hz from its carrier shift
    (``LineList.harmonic``).

    Final sites. The lower state may end in its well of the atom's own site
    (k = 0) or of a neighbouring one (k = +/-1; towards theta = +/-pi all of
    the carrier is on k = +/-1). Those final states are orthogonal and, in
    the periodic lattice, degenerate (``spectrum`` puts their lines at the
    same frequency), so the initial state couples to one bright combination
    of them with s^2 = sum_k s_k^2: each line here carries the coupling
    summed in quadrature over the sites (per axis, the Franck-Condon matrix
    summed over k), which is the exact two-level dynamics for degenerate
    final states (not the amplitude sum, and not independent lines, which
    ``combine_lines`` would treat as successive transfers).
    """
    axes = list(sp["axes"])
    data = [sp["axes"][ax] for ax in axes]
    amp = [np.sqrt(np.clip(_franck_condon_all_sites(d), 0.0, None)) for d in data]     # (n_lower, n_upper)
    harmonic = any("levels" not in d for d in data)
    if harmonic:
        share = float(sp["carrier_shift_hz"]) / len(axes)
        eu = [share + d["f_upper_hz"] * np.arange(a.shape[1]) for d, a in zip(data, amp)]
        el = [d["f_lower_hz"] * np.arange(a.shape[0]) for d, a in zip(data, amp)]
    else:
        if states is None:
            states = tuple(data[0]["levels"].keys())[:2]
        upper, lower = states
        eu = [d["levels"][upper]["energies_hz"] for d in data]
        el = [d["levels"][lower]["energies_hz"] for d in data]
    weights = [np.asarray(d["thermal"], float) for d in data]
    pop = weights[0]
    for w in weights[1:]:
        pop = np.multiply.outer(pop, w)
    flat = pop.ravel()
    keep = np.nonzero(flat >= min_population)[0]
    if max_states is not None and keep.size > max_states:
        keep = keep[np.argsort(flat[keep])[::-1][:max_states]]
    if keep.size == 0:
        keep = np.array([int(np.argmax(flat))])
    idx = np.unravel_index(np.sort(keep), pop.shape)
    p0 = pop[idx]
    ids = np.arange(p0.size)
    diag_amp = [amp[a][idx[a], idx[a]] for a in range(len(axes))]
    diag_e = [eu[a][idx[a]] - el[a][idx[a]] for a in range(len(axes))]
    carrier_s = np.prod(diag_amp, axis=0)
    carrier_f = np.sum(diag_e, axis=0)
    det, sc, st, lab_ax, lab_dn = [carrier_f], [carrier_s], [ids], [np.full(p0.size, "")], [np.zeros(p0.size, int)]
    for a, ax in enumerate(axes):
        n = idx[a]
        others_s = np.ones(p0.size)
        for b in range(len(axes)):
            if b != a:
                others_s = others_s * diag_amp[b]
        others_f = carrier_f - diag_e[a]
        n_low = el[a].size
        for dn in range(-max_dn, max_dn + 1):
            if dn == 0:
                continue
            n2 = n + dn
            ok = (n2 >= 0) & (n2 < n_low)
            if not ok.any():
                continue
            det.append(others_f[ok] + eu[a][n[ok]] - el[a][n2[ok]])
            sc.append(others_s[ok] * amp[a][n2[ok], n[ok]])
            st.append(ids[ok])
            lab_ax.append(np.full(int(ok.sum()), ax))
            lab_dn.append(np.full(int(ok.sum()), dn))
    state = np.concatenate(st)
    out = make_lines(np.concatenate(det), np.concatenate(sc), p0[state], state, np.concatenate(lab_ax),
                     np.concatenate(lab_dn), float(p0.sum()))
    out.harmonic = harmonic
    return out


def strength_outside_lines(lines: LineList):
    """Fraction of the kept initial states' line strength that no listed line
    carries: 1 - sum p s^2 / kept_population. Every initial state's
    strength over all final states is 1 (completeness of the lower state's
    levels and sites), so this is what the line list leaves out: sidebands
    on two axes at once, |dn| > max_dn, final levels above the level cut and
    sites not solved. 0 for a list without lines."""
    if lines.kept_population <= 0 or len(lines) == 0:
        return 0.0
    inside = float(np.sum(lines.population * lines.scale ** 2)) / lines.kept_population
    return float(max(0.0, 1.0 - inside))


def power_scan(lines: LineList, pulse: Pulse, f_set_hz, power_factors, spread_hz=0.0, reach_hz=None,
               progress=None, cancelled=None):
    """Transferred population at one set frequency versus power, power_factors =
    power / the pulse's power (the Rabi frequency scales as their square root).
    Direct propagation of every line (``combine_lines``); lines farther than
    reach_hz (default: the response reach at the highest power) are skipped,
    lines with near-identical (s, detuning) propagated once."""
    factors = np.sqrt(np.asarray(power_factors, float))
    top = pulse.with_rabi(pulse.rabi_hz * float(factors.max()))
    d = f_set_hz - lines.detuning_hz
    if reach_hz is None:
        reach_hz = response_reach_hz(top, float(np.max(np.abs(d))) + 1.0) if pulse.shape != "square" else np.inf
    near = (np.abs(d) <= reach_hz + SPREAD_SIGMAS * spread_hz) & (lines.population > 0)
    if not near.any():
        return np.zeros(factors.size)
    sel = lines.select(near)
    d = d[near]
    d_res = 1.0 / (40 * pulse.duration_s)
    key = np.stack([np.round(sel.scale / 1e-4), np.round(d / d_res)], axis=1)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    s_u, d_u = uniq[:, 0] * 1e-4, uniq[:, 1] * d_res
    nodes = [(0.0, 1.0)]
    if spread_hz > 0:
        x, w = np.polynomial.hermite.hermgauss(SPREAD_NODES)
        nodes = list(zip(np.sqrt(2) * spread_hz * x, w / np.sqrt(np.pi)))
    p_u = np.zeros((s_u.size, factors.size))
    for i, (dx, wx) in enumerate(nodes):
        p = _propagate(pulse, s_u[:, None] * factors[None, :], (d_u + dx)[:, None], cancelled=cancelled)
        if p is None:
            return None
        p_u += wx * p
        if progress is not None:
            progress(i + 1, len(nodes))
    return combine_lines(sel, p_u[inverse])


def response_width(detuning_hz, p):
    """(peak, full width at half maximum in Hz) of a single-line response curve;
    the width spans the outermost half-maximum crossings."""
    d, p = np.asarray(detuning_hz, float), np.asarray(p, float)
    peak = float(p.max())
    if peak <= 0:
        return 0.0, 0.0
    half = 0.5 * peak
    above = np.nonzero(p >= half)[0]
    i, j = above[0], above[-1]
    left = d[0] if i == 0 else float(np.interp(half, [p[i - 1], p[i]], [d[i - 1], d[i]]))
    right = d[-1] if j == p.size - 1 else float(np.interp(half, [p[j + 1], p[j]], [d[j + 1], d[j]]))
    return peak, float(right - left)


# ========================================================= coupled reference
def coupled_transfer(e_upper_hz, e_lower_hz, overlap, populations, pulse: Pulse, f_set_hz):
    """Multi-level reference for one vibrational axis: all |upper, n> and
    |lower, m> driven together (no independent-line approximation).

    e_upper_hz, e_lower_hz: level energies relative to the free-atom line
    (a line n -> m lies at e_upper[n] - e_lower[m]); overlap: signed
    <lower m|upper n>, shape (n_lower, n_upper); populations of the upper
    levels. Returns the transferred population for each set frequency.
    Piecewise-constant propagation over the same steps as ``transfer``,
    each step's Hamiltonian diagonalized exactly (stacked over frequencies)."""
    eu, el = np.asarray(e_upper_hz, float), np.asarray(e_lower_hz, float)
    o = np.asarray(overlap, float)
    nu, nl = eu.size, el.size
    f = np.atleast_1d(np.asarray(f_set_hz, float))
    dt, amp, off = _steps(pulse)
    u = np.broadcast_to(np.eye(nu + nl, dtype=complex), (f.size, nu + nl, nu + nl)).copy()
    h = np.zeros((f.size, nu + nl, nu + nl))
    h[:, nu + np.arange(nl), nu + np.arange(nl)] = el
    for k in range(dt.size):
        h[:, np.arange(nu), np.arange(nu)] = eu[None, :] - (f[:, None] + off[k])
        h[:, nu:, :nu] = 0.5 * pulse.rabi_hz * amp[k] * o
        h[:, :nu, nu:] = 0.5 * pulse.rabi_hz * amp[k] * o.T
        lam, vec = np.linalg.eigh(h)
        u = (vec * np.exp(-2j * np.pi * lam * dt[k])[:, None, :]) @ np.swapaxes(vec, 1, 2) @ u
    transferred = np.sum(np.abs(u[:, nu:, :nu]) ** 2, axis=1)          # (n_f, n_upper)
    return transferred @ np.asarray(populations, float)
