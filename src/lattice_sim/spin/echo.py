"""Spin-echo sequences with state-dependent transport: phases, noise, contrast.

Model. The qubit is stored in |3,-1> / |4,+1>. For each transport the two
storage states are mapped onto the transport partners |3,-m> / |3,+m>, the
lattice polarization is rotated (theta(t)), and the states are mapped back.
Between the pi/2 pulses the relative phase is

    phi = 2 pi  integral  s(t) * delta(t) dt,

with s(t) = +/-1 the toggling sign (flipped by every pi pulse) and delta(t)
the instantaneous differential frequency. delta is a sum of channels, each
a known coefficient c_k(t) (Hz per unit of noise) times a noise n_k(t):

    field      c = kappa_B in transport states, ~0 in storage   n = dB (mG)
    intensity  c = 2 dE0 a^2 + S eta a (transport), LS_storage/2 (storage)
                                                                n = dI/I
    ellipticity c = S a                                          n = d eta
    rotation   c = d(Delta E)/d theta                            n = d theta (rad)

where a is the local intensity relative to nominal (site dependence from
the beam envelope). Random noise enters through its power spectral density
(filter functions; checked by Monte Carlo); leg-to-leg systematic changes
(the EOM thermal transient) enter deterministically and dephase through
their site dependence.

Time is in microseconds on a uniform grid; frequencies in Hz.
"""

from dataclasses import dataclass, field

import numpy as np

AXIS_SCALE = {"x": 1.0, "y": (8.0 / 9.5) ** 2, "z": (6.25 / 9.5) ** 2}


# ---------------------------------------------------------------- sequences
@dataclass
class Transport:
    """One transport: ramp theta 0 -> theta_hold, hold, ramp back."""
    ramp_t_us: np.ndarray
    ramp_theta: np.ndarray
    hold_us: float = 600.0
    axis: str = "x"

    @property
    def duration_us(self):
        return 2 * (self.ramp_t_us[-1] - self.ramp_t_us[0]) + self.hold_us

    def theta(self, tau_us):
        """theta at time tau within this transport."""
        r = self.ramp_t_us[-1] - self.ramp_t_us[0]
        tau = np.asarray(tau_us, float)
        up = np.interp(tau, self.ramp_t_us - self.ramp_t_us[0], self.ramp_theta)
        down = np.interp(self.duration_us - tau, self.ramp_t_us - self.ramp_t_us[0], self.ramp_theta)
        return np.where(tau <= r, up, np.where(tau >= r + self.hold_us, down, self.ramp_theta[-1]))


@dataclass
class Sequence:
    """Ordered segments: ('pi',), ('wait', us), ('transport', Transport).

    The initial and final pi/2 pulses are implicit. The map pulses into and
    out of the transport states are treated as instantaneous and ideal here;
    their errors are budgeted separately.
    """
    segments: list
    dt_us: float = 5.0
    label: str = ""

    def build(self):
        t, s, theta, axis_scale, leg = [], [], [], [], []
        now, sign, leg_index = 0.0, 1.0, 0
        for seg in self.segments:
            kind = seg[0]
            if kind == "pi":
                sign = -sign
                continue
            if kind == "wait":
                n = int(round(seg[1] / self.dt_us))
                tt = now + self.dt_us * (np.arange(n) + 0.5)
                t.append(tt), s.append(np.full(n, sign)), theta.append(np.full(n, np.nan))
                axis_scale.append(np.zeros(n)), leg.append(np.full(n, -1))
                now += n * self.dt_us
            elif kind == "transport":
                tr = seg[1]
                n = int(round(tr.duration_us / self.dt_us))
                tau = self.dt_us * (np.arange(n) + 0.5)
                t.append(now + tau), s.append(np.full(n, sign)), theta.append(tr.theta(tau))
                axis_scale.append(np.full(n, AXIS_SCALE[tr.axis])), leg.append(np.full(n, leg_index))
                leg_index += 1
                now += n * self.dt_us
        cat = np.concatenate
        return SequenceGrid(cat(t), cat(s), cat(theta), cat(axis_scale), cat(leg), self.dt_us)


@dataclass
class SequenceGrid:
    t_us: np.ndarray
    sign: np.ndarray
    theta: np.ndarray        # nan when in storage states
    axis_scale: np.ndarray   # depth relative to x (0 in storage)
    leg: np.ndarray          # transport index, -1 in storage
    dt_us: float

    @property
    def in_transport(self):
        return ~np.isnan(self.theta)


# ---------------------------------------------------------- channel coefficients
def coefficients(grid, pair, eta_of_theta, a=1.0):
    """Coefficient arrays c_k(t) (Hz per unit noise) for a site of relative
    intensity a. eta_of_theta: callable giving the ellipticity vs theta."""
    tr = grid.in_transport
    th = np.where(tr, grid.theta, 0.0)
    sc = grid.axis_scale * a                       # light-shift scale at this site
    eta = np.where(tr, eta_of_theta(th), 0.0)
    e0, s = pair.de0(th), pair.s(th)
    c = {
        "field": np.where(tr, pair.kappa_b_hz_per_mg, 0.0),
        "intensity": np.where(tr, 2 * e0 * sc**2 + s * eta * sc, 0.5 * pair.storage_ls_hz * a),
        "ellipticity": np.where(tr, s * sc, 0.0),
        "rotation": np.where(tr, pair.ddtheta(th, eta) * np.where(tr, sc, 0) * 1.0, 0.0),
        "static": np.where(tr, e0 * sc**2 + s * eta * sc, pair.storage_ls_hz * a),
    }
    return c


def phase(grid, coeff, noise):
    """2 pi * integral s c n dt for noise samples n (array or scalar)."""
    return 2 * np.pi * np.sum(grid.sign * coeff * noise) * grid.dt_us * 1e-6


# ------------------------------------------------------------- noise spectra
@dataclass
class NoiseSpectrum:
    """One-sided PSD: white + 1/f + discrete lines (random phase), units^2/Hz.

    white: level (units^2/Hz). flicker: S = flicker / f for f >= f_min.
    lines: list of (frequency Hz, rms amplitude units).
    """
    white: float = 0.0
    flicker: float = 0.0
    f_min: float = 0.1
    lines: list = field(default_factory=list)

    def psd(self, f):
        f = np.asarray(f, float)
        return self.white + np.where(f >= self.f_min, self.flicker / np.maximum(f, 1e-12), 0.0)

    def scaled(self, k):
        return NoiseSpectrum(self.white * k**2, self.flicker * k**2, self.f_min,
                             [(f, a * k) for f, a in self.lines])


def filter_variance(grid, coeff, spectrum, f_max=None):
    """Phase variance (rad^2) from a stationary noise with the given PSD."""
    y = grid.sign * coeff
    n = y.size
    pad = 1 << int(np.ceil(np.log2(4 * n)))
    dt = grid.dt_us * 1e-6
    g = np.fft.rfft(y, pad) * dt                     # G(f) = int y e^{-i 2 pi f t} dt
    f = np.fft.rfftfreq(pad, dt)
    df = f[1] - f[0]
    sel = slice(1, None) if f_max is None else (f > 0) & (f <= f_max)
    var = (2 * np.pi) ** 2 * np.sum(spectrum.psd(f[sel]) * np.abs(g[sel]) ** 2) * df
    tt = grid.t_us * 1e-6
    for fl, amp in spectrum.lines:                    # random-phase line: <n^2> = amp^2
        gl = np.sum(y * np.exp(-2j * np.pi * fl * tt)) * dt
        var += (2 * np.pi) ** 2 * amp**2 * np.abs(gl) ** 2 / 2 * 2  # |G|^2 * 2 * (amp^2/2)
    return var


def sample_noise(grid, spectrum, rng, n_samples=1):
    """Monte Carlo noise traces with the given PSD (Gaussian, random-phase lines)."""
    n = grid.t_us.size
    dt = grid.dt_us * 1e-6
    pad = 1 << int(np.ceil(np.log2(4 * n)))
    f = np.fft.rfftfreq(pad, dt)
    amp = np.sqrt(spectrum.psd(f) * (f[1] - f[0]) / 4)  # x = 2 Re(z): var 4 amp^2 = S df
    amp[0] = 0.0
    traces = []
    for _ in range(n_samples):
        z = amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
        x = np.fft.irfft(z, pad) * pad
        x = x[:n]
        tt = grid.t_us * 1e-6
        for fl, a_rms in spectrum.lines:
            x = x + np.sqrt(2) * a_rms * np.cos(2 * np.pi * fl * tt + rng.uniform(0, 2 * np.pi))
        traces.append(x)
    return np.array(traces)


# --------------------------------------------------------- systematic leg drift
@dataclass
class LegTransient:
    """EOM state relaxing after each transport (thermal transient).

    After the end of a transport, the next transport sees a rotation offset
    and an ellipticity change d = A * exp(-gap / tau), with gap the time since
    the previous transport ended. The first transport sees none.
    """
    d_theta_rad: float = 0.0
    d_eta: float = 0.0
    tau_ms: float = 50.0

    def offsets(self, grid):
        """Per-time-step (d_theta, d_eta) arrays. Each axis has its own EOM,
        so the transient of a transport depends only on the previous
        transport of the same axis."""
        dth = np.zeros_like(grid.t_us)
        deta = np.zeros_like(grid.t_us)
        legs = [k for k in np.unique(grid.leg) if k >= 0]
        prev_end = {}
        for k in legs:
            idx = np.flatnonzero(grid.leg == k)
            axis = float(grid.axis_scale[idx[0]])
            if axis in prev_end:
                gap_ms = (grid.t_us[idx[0]] - prev_end[axis]) * 1e-3
                w = np.exp(-gap_ms / self.tau_ms)
                dth[idx] = self.d_theta_rad * w
                deta[idx] = self.d_eta * w
            prev_end[axis] = grid.t_us[idx[-1]]
        return dth, deta


# ------------------------------------------------------------------- budget
@dataclass
class EchoModel:
    pair: object
    eta_of_theta: object
    field: NoiseSpectrum
    intensity: NoiseSpectrum
    transient: LegTransient = field(default_factory=LegTransient)
    leg_intensity_sigma: float = 0.0     # shot-to-shot, leg-to-leg dI/I (quasi-static per leg)
    site_a: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    scatter_rate_per_s: float = 0.0      # coherence-destroying scattering, total
    motional_contrast: float = 1.0        # from the transport engine

    def analyse(self, sequence):
        g = sequence.build()
        c1 = coefficients(g, self.pair, self.eta_of_theta, 1.0)
        out = {"duration_ms": g.t_us[-1] * 1e-3 + g.dt_us * 1e-3 / 2,
               "transport_ms": np.sum(g.in_transport) * g.dt_us * 1e-3}
        # random noise (common to all sites to first order)
        v_field = filter_variance(g, c1["field"], self.field)
        v_int = filter_variance(g, c1["intensity"], self.intensity)
        # quasi-static leg-to-leg intensity: each transport its own dI/I
        legs = [k for k in np.unique(g.leg) if k >= 0]
        v_leg = 0.0
        for k in legs:
            sel = g.leg == k
            v_leg += (2 * np.pi * np.sum((g.sign * c1["intensity"])[sel]) * g.dt_us * 1e-6) ** 2
        v_leg *= self.leg_intensity_sigma**2
        # systematic transient, evaluated per site
        dth, deta = self.transient.offsets(g)
        site_phase = []
        for a in self.site_a:
            ca = coefficients(g, self.pair, self.eta_of_theta, a)
            site_phase.append(phase(g, ca["rotation"], dth) + phase(g, ca["ellipticity"], deta)
                              + phase(g, ca["static"], 1.0))
        site_phase = np.array(site_phase)
        mean_vec = np.mean(np.exp(1j * site_phase))
        out.update(
            phase_rad=float(np.angle(mean_vec)),
            sigma_field=float(np.sqrt(v_field)),
            sigma_intensity=float(np.sqrt(v_int)),
            sigma_leg_drift=float(np.sqrt(v_leg)),
            c_field=float(np.exp(-v_field / 2)),
            c_intensity=float(np.exp(-v_int / 2)),
            c_leg_drift=float(np.exp(-v_leg / 2)),
            c_inhomogeneous=float(np.abs(mean_vec)),
            c_scatter=float(np.exp(-self.scatter_rate_per_s * out["duration_ms"] * 1e-3)),
            c_motion=float(self.motional_contrast),
        )
        out["contrast"] = float(np.prod([out[k] for k in out if k.startswith("c_")]))
        return out


def site_intensities(half_extent_um=10.0, waist_um=60.0, spacing_um=4.84):
    """Relative x-lattice intensity at occupied sites (Gaussian envelope in the
    two directions transverse to the x-pair beams), normalized to the centre."""
    n = int(np.floor(half_extent_um / spacing_um))
    c = np.arange(-n, n + 1) * spacing_um
    xx, zz = np.meshgrid(c, c)
    return np.exp(-2 * (xx**2 + zz**2) / waist_um**2).ravel()
