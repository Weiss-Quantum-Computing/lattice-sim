"""Site geometry and lattice fillings for the cluster-state model.

Sites of an nx x ny x nz block of the simple-cubic lattice are indexed
``i = x + nx * (y + ny * z)`` (the convention of Weiss-Quantum-Computing/
Cluster, ``Cluster3D.coordinate_to_index_3d``). z is the imaging axis, so a
"plane" in the lab sense is a fixed z. An occupancy map is a flat boolean
array of length nx * ny * nz.

The fillings follow the experiment plan (Du 2025, Sec. 4.2.1): parity
loading gives a random ~50 % filling (Du quotes 54 % in practice); sorting
gives a compact block (4 x 4 x 3 or 5 x 5 x 2) at 95-97 % filling; atoms left
next to the sorted block ("peripheral atoms") collide with it unless cleared.
Lines and single planes are the 1D and 2D test configurations.
"""

from dataclasses import dataclass

import numpy as np

AXES = "xyz"
PARITY_LOADING_FILL = 0.54        # Du p. 30: parity loading, nominally 50 %
SORTED_FILL_443 = 0.95            # Du p. 6: 4 x 4 x 3 after three sorts
SORTED_FILL_552 = 0.97            # Du p. 6: 5 x 5 x 2 after three sorts


@dataclass(frozen=True)
class Region:
    nx: int
    ny: int
    nz: int

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def n(self):
        return self.nx * self.ny * self.nz

    def index(self, x, y, z):
        return x + self.nx * (y + self.ny * z)

    def coords(self, i):
        i = int(i)
        return i % self.nx, (i // self.nx) % self.ny, i // (self.nx * self.ny)

    def all_coords(self):
        """(n, 3) integer array of (x, y, z) in index order."""
        i = np.arange(self.n)
        return np.stack([i % self.nx, (i // self.nx) % self.ny, i // (self.nx * self.ny)], axis=1)

    def parity(self):
        """Checkerboard sublattice label (x + y + z) mod 2 of every site."""
        return self.all_coords().sum(axis=1) % 2


def _axis(a):
    return AXES.index(a) if isinstance(a, str) else int(a)


def bonds(region, occupancy, axis):
    """Neighbour pairs (r, s) along one axis with both sites occupied; s = r + e_axis.

    In a collision leg the |3,+m> component of the atom at r moves +e and meets
    the |3,-m> component of the atom at s (chain.py convention, the repo's sign
    convention for x, applied to every axis).
    """
    ax = _axis(axis)
    occ = np.asarray(occupancy, bool)
    c = region.all_coords()
    out = []
    for r in np.flatnonzero(occ):
        s_c = c[r].copy()
        s_c[ax] += 1
        if s_c[ax] < region.shape[ax]:
            s = region.index(*s_c)
            if occ[s]:
                out.append((int(r), int(s)))
    return out


def neighbours(region, occupancy, axes="xyz"):
    """Occupied nearest neighbours of every site along the given axes (list of lists)."""
    nb = [[] for _ in range(region.n)]
    for a in axes:
        for r, s in bonds(region, occupancy, a):
            nb[r].append(s)
            nb[s].append(r)
    return nb


def degrees(region, occupancy, axes="xyz"):
    return np.array([len(v) for v in neighbours(region, occupancy, axes)], int)


# ------------------------------------------------------------ fillings
def empty(region):
    return np.zeros(region.n, bool)


def random_filling(region, fraction=PARITY_LOADING_FILL, rng=None):
    rng = np.random.default_rng(rng)
    return rng.random(region.n) < fraction


def block(region, size, origin=None):
    """A unit-filled block of the given size; centred (rounded down) when origin is None."""
    size = tuple(int(s) for s in size)
    if origin is None:
        origin = tuple((region.shape[k] - size[k]) // 2 for k in range(3))
    occ = empty(region)
    c = region.all_coords()
    inside = np.all((c >= np.array(origin)) & (c < np.array(origin) + np.array(size)), axis=1)
    occ[inside] = True
    return occ


def remove_random(occupancy, k=None, fraction=None, rng=None):
    """Remove k atoms (or each atom with probability 1 - fraction) at random."""
    rng = np.random.default_rng(rng)
    occ = np.array(occupancy, bool)
    filled = np.flatnonzero(occ)
    if k is not None:
        drop = rng.choice(filled, size=min(int(k), filled.size), replace=False)
    else:
        drop = filled[rng.random(filled.size) >= float(fraction)]
    occ[drop] = False
    return occ


def peripheral(region, occupancy, probability, rng=None):
    """Add atoms on empty sites face-adjacent to the filled set, each with the given probability."""
    rng = np.random.default_rng(rng)
    occ = np.array(occupancy, bool)
    c = region.all_coords()
    shell = []
    for i in np.flatnonzero(~occ):
        for ax in range(3):
            for d in (-1, 1):
                q = c[i].copy()
                q[ax] += d
                if 0 <= q[ax] < region.shape[ax] and occ[region.index(*q)]:
                    shell.append(i)
                    break
            else:
                continue
            break
    shell = np.array(sorted(set(shell)), int)
    occ[shell[rng.random(shell.size) < probability]] = True
    return occ


def lines(region, axis, positions):
    """Full lines along `axis` at the given transverse positions [(u, v), ...]."""
    ax = _axis(axis)
    others = [k for k in range(3) if k != ax]
    occ = empty(region)
    c = region.all_coords()
    for u, v in positions:
        occ[(c[:, others[0]] == u) & (c[:, others[1]] == v)] = True
    return occ


def plane(region, normal="z", index=0):
    ax = _axis(normal)
    occ = empty(region)
    occ[region.all_coords()[:, ax] == int(index)] = True
    return occ
