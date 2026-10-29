"""Molasses beam geometries.

Each geometry is a list of beams (unit wave vector k_hat, complex unit
polarization eps, relative intensity). Phases are not part of the geometry:
the six beams are not interferometrically locked, and the lattice spacing
(4.84 um) is incommensurate with the molasses pattern, so each lattice site
sees an independent random phase configuration. The liquid-crystal phase
ramps on two pairs (Wu 2020 App. B; Kumar 2019 Sec. 5.3.1) sweep it in time
on a 30 ms scale, slower than the cooling time; the model therefore draws
one random set of six phases per atom and keeps it fixed for the run, and
ensemble averages over atoms stand for averages over sites and time.

LAB_BREWSTER (Li 2009, Sec. 2.3.2, Fig. 2-9; status: thesis, not re-verified):
three beams enter through the cell bottom at Brewster's angle (55.4 deg
from the vertical), p-polarized (in the plane of incidence), and are
retro-reflected from above through a quarter-wave plate double pass, so each
pair is lin-perp-lin with the return beam s-polarized. Seen from the top,
pair "4" is parallel to the long side of the cell and pairs "5" and "6" are
perpendicular to each other at 45 deg to the cell. Li gives the 3D angles
only for the MOT beams (67 deg from the vertical; pairs 2-3: 81 deg, pair 1 to
2 or 3: 117 deg), which fix the MOT azimuths at 45, 135 and 270 deg in the
cell frame; the molasses pairs are taken to follow the same pattern at the
Brewster angle (azimuths 270, 45, 135 deg for pairs 4, 5, 6: 71 deg between
pairs 5 and 6, 99 deg between pair 4 and either). The cell frame is assumed
to coincide with the lattice/bias-coil x, y axes (AZIMUTH_OFFSET_DEG = 0);
neither the azimuths nor that alignment is stated for the molasses in any
thesis (data request 37).
"""

import numpy as np

BREWSTER_DEG = 55.4                        # fused silica, Li 2009 p. 24
AZIMUTHS_DEG = (270.0, 45.0, 135.0)        # pairs 4, 5, 6 relative to lattice x (assumed)
AZIMUTH_OFFSET_DEG = 0.0


def _pair_brewster(theta, phi):
    k = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_p = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_s = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return [(k, e_p.astype(complex)), (-k, e_s.astype(complex))]


def lab_brewster(azimuth_offset_deg=AZIMUTH_OFFSET_DEG, polar_deg=BREWSTER_DEG,
                 azimuths_deg=AZIMUTHS_DEG, imbalance=None):
    """Six beams of the lab molasses; ``imbalance`` optionally gives 6 relative intensities."""
    beams = []
    th = np.deg2rad(polar_deg)
    for az in azimuths_deg:
        beams += _pair_brewster(th, np.deg2rad(az + azimuth_offset_deg))
    return _pack(beams, imbalance)


def orthogonal_linperp(imbalance=None):
    """Three orthogonal lin-perp-lin pairs (Salomon et al. 1990, EPL 12, 683)."""
    ex, ey, ez = np.eye(3)
    beams = [(ex, ey), (-ex, ez), (ey, ez), (-ey, ex), (ez, ex), (-ez, ey)]
    return _pack([(k, e.astype(complex)) for k, e in beams], imbalance)


def one_d_linperp():
    """1D lin-perp-lin along z (Dalibard & Cohen-Tannoudji 1989)."""
    ex, ey, ez = np.eye(3)
    return _pack([(ez, ex.astype(complex)), (-ez, ey.astype(complex))], None)


def _pack(beams, imbalance):
    k = np.array([b[0] for b in beams], dtype=float)
    eps = np.array([b[1] / np.linalg.norm(b[1]) for b in beams], dtype=complex)
    rel = np.ones(len(beams)) if imbalance is None else np.asarray(imbalance, dtype=float)
    for kk, ee in zip(k, eps):
        assert abs(np.dot(kk, ee)) < 1e-12, "polarization not transverse"
    return dict(k_hat=k, eps=eps, rel_intensity=rel)


GEOMETRIES = {"lab (Brewster lin⊥lin pairs)": lab_brewster,
              "orthogonal lin⊥lin": orthogonal_linperp,
              "1D lin⊥lin": one_d_linperp}
