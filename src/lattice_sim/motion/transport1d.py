"""One-dimensional wave-packet transport along a lattice axis.

The atom evolves under H(t) = p^2 / 2m + V(x, theta(t)) for one internal
state, with theta(t) the rotation angle of the EOM arm. Propagation is the
second-order split-step Fourier method on a uniform periodic grid, which is
exact for any potential shape up to O(dt^3) per step; the grid only needs to
contain the one lattice site the atom occupies.

Excitation is scored in the lab frame, as in Transport-Optimizer: P0(t) is
the population of the instantaneous ground state of V(x, theta(t)), so a
packet lagging behind a moving trap counts as excited. The instantaneous
ground state is found from a banded fourth-order finite-difference
Hamiltonian on the same grid.

Potentials are supplied as a ``PotentialTable``: V tabulated on (theta, x),
interpolated linearly in theta. Two constructors are provided, one from the
first-principles model (``fields.potential.StatePotential``) and one from a
harmonic calibration (trap minimum and frequency versus theta), so the two
descriptions can be compared on identical numerics.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig_banded

from ..constants import H, HBAR, MASS_CS


@dataclass
class PotentialTable:
    thetas: np.ndarray       # (n_theta,), rad, increasing
    x: np.ndarray            # (n_x,), m, uniform
    values_hz: np.ndarray    # (n_theta, n_x), V / h in Hz

    def at(self, theta):
        """V(x) / h at angle theta, linear interpolation between rows."""
        th = self.thetas
        theta = float(np.clip(theta, th[0], th[-1]))
        i = int(np.clip(np.searchsorted(th, theta) - 1, 0, th.size - 2))
        w = (theta - th[i]) / (th[i + 1] - th[i])
        return (1 - w) * self.values_hz[i] + w * self.values_hz[i + 1]

    @classmethod
    def from_model(cls, model, state, thetas, x):
        values = np.empty((len(thetas), len(x)))
        for k, theta in enumerate(thetas):
            values[k] = model.potential(state, x, theta)
        return cls(np.asarray(thetas, float), np.asarray(x, float), values)

    def save(self, path):
        np.savez_compressed(path, thetas=self.thetas, x=self.x, values_hz=self.values_hz)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["thetas"], data["x"], data["values_hz"])

    @classmethod
    def harmonic(cls, center_of_theta, frequency_hz_of_theta, thetas, x):
        """V = (1/2) m omega(theta)^2 (x - x0(theta))^2, a pure harmonic trap."""
        thetas = np.asarray(thetas, float)
        x0 = np.asarray(center_of_theta(thetas), float)[:, None]
        omega = 2 * np.pi * np.asarray(frequency_hz_of_theta(thetas), float)[:, None]
        values = 0.5 * MASS_CS * omega**2 * (x[None, :] - x0) ** 2 / H
        return cls(thetas, np.asarray(x, float), values)


def uniform_grid(x_low, x_high, n):
    """Periodic grid suitable for the FFT (endpoint excluded)."""
    return np.linspace(x_low, x_high, n, endpoint=False)


def grid_for_direction(sign, n=1024, behind_m=0.9e-6, ahead_m=3.3e-6):
    """Grid spanning one lattice site's full path in the direction of motion.

    Its length (4.2 um) is shorter than the 4.84 um spacing, so it never
    contains two sites, and it extends 0.9 um behind the start and beyond
    the end (half a site, 2.42 um) to hold the wave packet's tails.
    """
    if sign >= 0:
        return uniform_grid(-behind_m, ahead_m, n)
    return uniform_grid(-ahead_m, behind_m, n)


class Transport1D:
    """Split-step propagator for one internal state on a potential table."""

    def __init__(self, table: PotentialTable):
        self.table = table
        x = table.x
        self.dx = float(x[1] - x[0])
        self.n = x.size
        k = 2 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.kinetic_hz = HBAR * k**2 / (2 * MASS_CS) / (2 * np.pi)
        # Banded 4th-order finite-difference kinetic operator, in Hz
        c = HBAR / (2 * MASS_CS * self.dx**2) / (2 * np.pi)
        self._band = np.zeros((3, self.n))
        self._band[0, :] = c * 2.5             # diagonal
        self._band[1, :-1] = c * (-4.0 / 3.0)   # first off-diagonal
        self._band[2, :-2] = c * (1.0 / 12.0)   # second off-diagonal

    def ground_state(self, theta, half_window_m=0.8e-6):
        """Lowest eigenstate (normalized on the grid) and its energy / h.

        Solved on a window of +/- half_window_m around the potential minimum
        (the trap ground state is < 0.2 um wide), zero outside it.
        """
        v = self.table.at(theta)
        centre = int(np.argmin(v))
        half = int(np.ceil(half_window_m / self.dx))
        lo, hi = max(0, centre - half), min(self.n, centre + half + 1)
        band = self._band[:, lo:hi].copy()
        band[0] = band[0] + v[lo:hi]
        band[1, -1] = 0.0
        band[2, -2:] = 0.0
        energy, vector = eig_banded(band, lower=True, select="i", select_range=(0, 0))
        psi = np.zeros(self.n, complex)
        psi[lo:hi] = vector[:, 0]
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        return psi, float(energy[0])

    def propagate(self, t_us, theta, dt_us=0.25, sample_every_us=5.0, psi0=None,
                  keep_states=False):
        """Evolve through the ramp theta(t), given as samples (t_us, theta).

        Returns times (us), P0 at those times, the mean position (m), and
        optionally the wave functions.
        """
        t_us = np.asarray(t_us, float)
        theta = np.asarray(theta, float)
        total = t_us[-1] - t_us[0]
        n_steps = int(np.ceil(total / dt_us))
        dt = total / n_steps
        if psi0 is None:
            psi0, _ = self.ground_state(theta[0])
        psi = psi0.copy()
        kin = np.exp(-2j * np.pi * self.kinetic_hz * dt * 1e-6)
        sample_stride = max(1, int(round(sample_every_us / dt)))
        times, p0, mean_x, states = [], [], [], []

        def record(time_us, th):
            g, _ = self.ground_state(th)
            times.append(time_us)
            p0.append(abs(np.vdot(g, psi)) ** 2)
            mean_x.append(float(np.sum(self.table.x * np.abs(psi) ** 2)))
            if keep_states:
                states.append(psi.copy())

        record(t_us[0], theta[0])
        for step in range(n_steps):
            t_mid = t_us[0] + (step + 0.5) * dt
            v = self.table.at(np.interp(t_mid, t_us, theta))
            half = np.exp(-1j * np.pi * v * dt * 1e-6)   # exp(-i 2 pi V dt / 2)
            psi = half * np.fft.ifft(kin * np.fft.fft(half * psi))
            if (step + 1) % sample_stride == 0 or step == n_steps - 1:
                t_now = t_us[0] + (step + 1) * dt
                record(t_now, np.interp(t_now, t_us, theta))
        out = (np.array(times), np.array(p0), np.array(mean_x))
        return out + (np.array(states),) if keep_states else out


def load_waveform(path, v_full_scale=5400.0):
    """Read a Transport-Optimizer ramp; return t (us) and theta (rad)."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(data["time_us"], float)
    v = np.maximum.accumulate(np.asarray(data["voltage_V"], float))
    return t, np.pi * v / v_full_scale, v


def band_mask(voltage, high=3800.0, low=100.0):
    return (voltage >= high - 1e-9) | (voltage <= low + 1e-9)


def harmonic_parameters(table, fit_half_width_m=20e-9):
    """Trap minimum (m) and harmonic frequency (Hz) of each tabulated row.

    Quadratic fit to the tabulated V over +/- fit_half_width around the grid
    minimum. Used to separate anharmonic effects from changes in the
    harmonic parameters.
    """
    x = table.x
    dx = x[1] - x[0]
    half = max(2, int(round(fit_half_width_m / dx)))
    centres, freqs = [], []
    for row in table.values_hz:
        i = int(np.clip(np.argmin(row), half, x.size - half - 1))
        k = slice(i - half, i + half + 1)
        c2, c1, _ = np.polyfit(x[k] - x[i], row[k], 2)
        centres.append(x[i] - c1 / (2 * c2))
        freqs.append(np.sqrt(max(2 * c2 * H, 0.0) / MASS_CS) / (2 * np.pi))
    return np.array(centres), np.array(freqs)
