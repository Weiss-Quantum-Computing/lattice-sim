"""State-resolved lattice potentials from the full ground-state Hamiltonian.

At every position the 16 x 16 Hamiltonian

    H(x) = A I.J + mu_B (g_J J + g_I I).B_q
           + U_unit [ sum_F r_F P_F |E(x)|^2 + r_v Im(E* x E)(x) . 2J ]

is diagonalized, so the vector light shift enters as a fictitious field that
adds to the bias field vectorially (the "drag" effect, Du Sec. 3.3.2) and
the quantization axis follows the total field, exactly rather than to first
order. r_F and r_v are the scalar and vector coefficients relative to the
centroid scalar coefficient (``atom.polarizability``). The potential of
|F, m_F> is its eigenvalue minus its bias-field-only energy; the state is
followed by maximal overlap with the bias-field eigenvector.

U_unit is fixed by a trap frequency: the scalar potential of the pure-linear,
theta = 0 lattice has curvature m omega^2 at its minimum.
"""

from dataclasses import dataclass, field

import numpy as np

from ..atom.ground_state import _operators, eigenlevels, ground_hamiltonian
from ..atom.polarizability import scalar_coefficient, vector_coefficient
from ..constants import H, MASS_CS


def _projectors():
    _, _, F = _operators()
    f_squared = sum(op @ op for op in F)
    p4 = (f_squared - 12 * np.eye(16)) / 8
    p3 = (20 * np.eye(16) - f_squared) / 8
    return {3: p3, 4: p4}


@dataclass
class StatePotential:
    pair: object
    b_field_gauss: np.ndarray
    trap_frequency_hz: float = 10e3
    include_hyperfine_scalar: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.b_field_gauss = np.asarray(self.b_field_gauss, float)
        wavelength = self.pair.wavelength_m
        s_c = scalar_coefficient(wavelength)
        self.r_v = vector_coefficient(wavelength) / s_c
        if self.include_hyperfine_scalar:
            self.r_f = {f: scalar_coefficient(wavelength, f) / s_c for f in (3, 4)}
        else:
            self.r_f = {3: 1.0, 4: 1.0}
        q = np.pi / self.pair.spacing_m
        omega = 2 * np.pi * self.trap_frequency_hz
        # Scalar |E|^2 = 2(1 - cos 2qx) near x = 0 -> U = U_unit 4 q^2 x^2
        self.u_unit_hz = MASS_CS * omega**2 / (8 * H * q**2)
        self._h0 = ground_hamiltonian(self.b_field_gauss)
        projectors = _projectors()
        self._scalar_op = sum(self.r_f[f] * projectors[f] for f in (3, 4))
        J, _, _ = _operators()
        self._sigma = [2 * j for j in J]
        self._bias = eigenlevels(self.b_field_gauss)

    def hamiltonians(self, x, theta):
        intensity, spin = self.pair.intensity_and_spin(x, theta)
        light = (
            intensity[:, None, None] * self._scalar_op
            + self.r_v * np.tensordot(spin, np.array(self._sigma), axes=1)
        )
        return self._h0[None] + self.u_unit_hz * light

    def potential(self, state, x, theta):
        """Potential energy / h (Hz) of |F, m_F> along x at rotation theta."""
        return self.potentials((state,), x, theta)[state]

    def potentials(self, states, x, theta):
        """{state: potential / h (Hz)} for several states from one diagonalization."""
        energies, vectors = np.linalg.eigh(self.hamiltonians(x, theta))
        rows = np.arange(energies.shape[0])
        out = {}
        for state in states:
            e_bias, v_bias = self._bias[state]
            overlaps = np.abs(np.einsum("i,nij->nj", v_bias.conj(), vectors)) ** 2
            index = np.argmax(overlaps, axis=1)
            if overlaps[rows, index].min() < 0.5:
                raise RuntimeError(f"state {state} is strongly mixed; labels unreliable")
            out[state] = energies[rows, index] - e_bias
        return out

    def minimum(self, state, theta, guess=0.0, half_window=None, n=801, fit_half_width=20e-9):
        """Trap minimum (m) and harmonic frequency (Hz) near ``guess``."""
        a = self.pair.spacing_m
        half_window = 0.3 * a if half_window is None else half_window
        x = np.linspace(guess - half_window, guess + half_window, n)
        v = self.potential(state, x, theta)
        i = int(np.argmin(v))
        if i in (0, n - 1):
            raise RuntimeError("minimum at window edge; widen the window")
        # Refine: quadratic fit over +/- fit_half_width around the coarse minimum
        x_fine = np.linspace(x[i] - fit_half_width, x[i] + fit_half_width, 41)
        v_fine = self.potential(state, x_fine, theta)
        c2, c1, _ = np.polyfit(x_fine - x[i], v_fine, 2)
        x_min = x[i] - c1 / (2 * c2)
        curvature = 2 * c2 * H  # J / m^2
        frequency = np.sqrt(max(curvature, 0.0) / MASS_CS) / (2 * np.pi)
        return x_min, frequency

    def track(self, state, thetas, start=0.0):
        """Follow the minimum continuously through a sequence of angles."""
        positions, frequencies = [], []
        guess = start
        for theta in thetas:
            x_min, f = self.minimum(state, theta, guess=guess)
            positions.append(x_min)
            frequencies.append(f)
            guess = x_min
        return np.array(positions), np.array(frequencies)
