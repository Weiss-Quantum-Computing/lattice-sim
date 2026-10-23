"""The |4,-4> -> |3,-3> spectrum at every site of a 9 x 9 x 9 block of the lattice.

Sites. Indices i, j, k = -GRID_HALF .. GRID_HALF along lab x, y, z; the
site position is r = (i a_x, j a_y, k a_z), with a_x the period of the x
pair along its fringe direction (lab x), a_y that of the y pair (lab y)
and a_z that of the z pair (lab z, the vertical imaging axis), each
lambda / (2 sin alpha) for the pair's crossing half-angle.

Model at a site. Each pair is described by ``beams.PairGeometry`` (arm
offsets, tilts, waist) in its own local lab frame (``PAIR_FRAMES``: rows =
fringe direction, propagation direction, static polarization); the site
position in that frame is ``PAIR_FRAMES[axis] @ r``, and
``beams.site_axis_config`` gives the pair's AxisConfig there: the field
amplitude of each arm at its distance from the arm's axis,
exp(-d^2 / w^2). So the Gaussian envelope of the beams is part of the
model: even with every arm centred, a corner site of the block (19 um
from the centre along two transverse directions) sees each x arm at
0.80-0.92 of its peak field, and the trap frequencies fall towards the
edges. The arms of a pair cross at +/- alpha, so along the bisector they
separate by 2 y sin(alpha) (3.4 um at y = 19 um): off-axis sites see the two
arms unbalanced, and through the ellipticity this moves the carrier
(``tests/test_grid.py``). The 1D-per-pair spectrum model of ``sidebands``
is then evaluated per site, with its limits: the pairs are independent 1D
problems, and the shear of the lattice under a vertical tilt (sites at
different heights at different x) does not enter (see ``beams``). The
waist is constant along the beams: the Rayleigh range pi w^2 / lambda is
13 mm for w = 60 um, so over +/- 20 um along a beam the beam radius
changes by 1e-6 relative (``rayleigh_waist_change``), which is neglected.

Speed. A pair's configuration at a site differs from its centre
configuration only in the two arm amplitudes (A_static, A_rotating):
spacing, frame, rotation, extinction ratio and depth are properties of the
pair. Its vibrational levels (``sidebands.axis_levels``: potentials,
energies, wavefunctions, well positions) are therefore computed exactly at
NODE_COUNT x NODE_COUNT Chebyshev nodes spanning the amplitudes on the
grid, and every number in them is interpolated (tensor-product polynomial,
wavefunction signs aligned between nodes) to each site's amplitudes. That
is 25 level calculations per pair instead of 729 (each arm's amplitude
depends on all three coordinates of a site, so there is no exact reduction
to the 81 transverse positions). The node calculations use
NODE_GRID_POINTS position points instead of the 1201 of ``SiteConfig``
(half the cost). ``sidebands.spectrum`` then builds each site's spectrum
from the interpolated levels (handed to it through its level cache), so
the grid follows that function exactly. Both shortcuts are checked against
the direct calculation in ``tests/test_grid.py``. A pair whose levels do
not have the same structure at every node (a different set of
neighbouring-site wells, an extended position grid) is computed directly
at every distinct site configuration instead (slow; reported in
``GridResult.direct_pairs``).

Results. ``GridResult`` holds the 729 site spectra (``spectrum``'s format,
'levels' without wavefunctions, grid and potentials: ``slim_levels``), the
positions and arm amplitudes; ``fit_plane`` is the 3D plane fit the lab's readstats ``planefit3d`` does on its site
maps, ``slice_means`` the per-slice averages of the lab's alignment
"slices" (sums over the other two axes), and ``ensemble_curve`` the
spectrum of the whole block (the mean of the site spectra, what a
frequency scan of the whole array records).
"""

import time
import traceback
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import fftconvolve

from ..constants import LATTICE_WAVELENGTH_M
from . import sidebands as sb
from .beams import NOMINAL_HALF_ANGLE, ArmGeometry, PairGeometry, site_axis_config
from .sidebands import LOWER, PAIR_FRAMES, UPPER, AxisConfig, SiteConfig, collapse_lines, line_shape

GRID_HALF = 4                    # sites -4 .. 4 along each lab axis: 9 x 9 x 9
NODE_COUNT = 5                   # Chebyshev nodes per arm amplitude (25 level calculations per pair)
NODE_GRID_POINTS = 601           # position points of a node's level calculation (SiteConfig: 1201); at 60
                                 # levels (z, 0.3 depth) line frequencies move < 6 Hz (tests/test_grid.py)
NODE_CACHE_BYTES = 300e6         # node level calculations kept between grids (~0.2 MB each at 13 levels)
WAIST_M = 60e-6                  # params/apparatus.yaml waist_um (thesis value, unmeasured at the atoms)
NOMINAL_SPACING_M = 4.8374e-6    # the spacing that AxisConfig.trap_frequency_hz refers to
SITE_BLOCK = 81                  # sites whose levels are interpolated in one matrix product
PAIRS = "xyz"
# a quantity that raises one of these at a site is undefined there (NaN, silently): a missing key,
# a vanishing carrier; any other exception is a bug in the quantity and is reported (first one)
QUANTITY_EXPECTED_ERRORS = (KeyError, IndexError, ZeroDivisionError, FloatingPointError)


def site_values(spectra, fn, out, sites):
    """out[m] = fn(spectra[m]) for m in sites (spectra None skipped). Expected
    failures (QUANTITY_EXPECTED_ERRORS) leave NaN; any other exception leaves
    NaN too and is counted. Returns (n_unexpected, first unexpected error as
    'Type: message', '' if none)."""
    n_bad, first = 0, ""
    for m in sites:
        sp = spectra[m]
        if sp is None:
            continue
        try:
            out[m] = float(fn(sp))
        except QUANTITY_EXPECTED_ERRORS:
            pass
        except Exception as err:                      # a bug in fn: keep going, but say so
            n_bad += 1
            if not first:
                first = "".join(traceback.format_exception_only(type(err), err)).strip()
    return n_bad, first


def rayleigh_waist_change(distance_m, waist_m=WAIST_M, wavelength_m=LATTICE_WAVELENGTH_M):
    """Relative change of a Gaussian beam's radius at ``distance_m`` from its focus."""
    z_r = np.pi * waist_m**2 / wavelength_m
    return float(np.sqrt(1 + (distance_m / z_r) ** 2) - 1)


# ------------------------------------------------------------------ pairs
@dataclass
class PairSetup:
    """One lattice pair across the grid: its depth/rotation/ER/handedness
    (``base``: trap frequency at NOMINAL_SPACING_M with unit amplitudes)
    and its beam geometry in its local lab frame."""
    base: AxisConfig
    geometry: PairGeometry

    @property
    def axis(self):
        return self.base.axis

    def spacing_m(self):
        return float(self.geometry.lattice()[0])

    def config_at(self, r_lab):
        """The exact AxisConfig of this pair at lab position r_lab."""
        r_local = PAIR_FRAMES[self.axis] @ np.asarray(r_lab, float)
        return site_axis_config(self.base, self.geometry, r_local, NOMINAL_SPACING_M)

    def amplitudes(self, positions):
        """(A_static, A_rotating) at every position (..., 3) in the lab frame."""
        pts = np.asarray(positions, float).reshape(-1, 3) @ PAIR_FRAMES[self.axis].T
        amp = np.array([self.geometry.amplitudes(r) for r in pts])
        shape = np.shape(positions)[:-1]
        return amp[:, 0].reshape(shape), amp[:, 1].reshape(shape)


def pair_setup(base, static_offset_m=0.0, rotating_offset_m=0.0, direction="pol", dalpha_rad=0.0,
               waist_m=WAIST_M):
    """PairSetup with the two arms' axes moved along the pair's static
    polarization ('pol') or along its fringes ('fringe'), and the crossing
    half-angle changed by dalpha_rad. ``base`` is the pair at nominal spacing
    and unit amplitudes (trap_frequency_hz = nominal x sqrt(depth))."""
    if direction not in ("pol", "fringe"):
        raise ValueError(f"direction must be 'pol' or 'fringe', not {direction!r}")
    key = "offset_pol_m" if direction == "pol" else "offset_fringe_m"
    geom = PairGeometry(ArmGeometry(**{key: float(static_offset_m)}, waist_m=waist_m),
                        ArmGeometry(**{key: float(rotating_offset_m)}, waist_m=waist_m),
                        half_angle_rad=NOMINAL_HALF_ANGLE + dalpha_rad)
    base = replace(base, spacing_m=NOMINAL_SPACING_M, static_amplitude=1.0, rotating_amplitude=1.0, frame=None)
    return PairSetup(base, geom)


def grid_positions(spacings_m, half=GRID_HALF):
    """Site positions (n, n, n, 3) in the lab frame, index order [i, j, k] = (x, y, z)."""
    idx = np.arange(-half, half + 1)
    ax, ay, az = (spacings_m[a] for a in PAIRS)
    i, j, k = np.meshgrid(idx, idx, idx, indexing="ij")
    return np.stack([i * ax, j * ay, k * az], axis=-1)


# -------------------------------------------------------- interpolation
def chebyshev_nodes(lo, hi, n):
    """n Chebyshev points of the first kind on [lo, hi] (one point if the span is empty)."""
    if hi - lo <= 1e-12 * max(1.0, abs(hi)) or n <= 1:
        return np.array([0.5 * (lo + hi)])
    k = np.arange(n)
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k + 1) * np.pi / (2 * n))[::-1]


def lagrange_weights(nodes, x):
    """Interpolation weights (len(x), len(nodes)) of the polynomial through the nodes."""
    nodes = np.asarray(nodes, float)
    x = np.atleast_1d(np.asarray(x, float))
    if nodes.size == 1:
        return np.ones((x.size, 1))
    w = np.ones((x.size, nodes.size))
    for m in range(nodes.size):
        for q in range(nodes.size):
            if q != m:
                w[:, m] *= (x - nodes[q]) / (nodes[m] - nodes[q])
    return w


def _leaves(obj, path=()):
    """(path, leaf) of a nested dict of arrays / numbers, in a fixed order."""
    if isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            yield from _leaves(obj[k], path + (k,))
    else:
        yield path, obj


def _set(tree, path, value):
    for k in path[:-1]:
        tree = tree.setdefault(k, {})
    tree[path[-1]] = value


class LevelTable:
    """One pair's ``axis_levels`` output at the nodes, interpolable: every
    float array or number that differs between nodes is interpolated, the
    rest (the position grid, integers) is shared. Raises ValueError when the
    nodes' levels do not have the same structure."""

    def __init__(self, nodes_s, nodes_r, levels):
        self.nodes_s, self.nodes_r = np.asarray(nodes_s), np.asarray(nodes_r)
        flat = [list(_leaves(lv)) for lv in levels]                     # node order: s outer, r inner
        ref = flat[0]
        paths = [p for p, _ in ref]
        for other in flat[1:]:
            if [p for p, _ in other] != paths:
                raise ValueError("the levels differ in structure between nodes")
        self.paths, self.constants, self.slots, self.aliases = paths, {}, [], {}
        columns, offset, first = [], 0, {}
        for m, path in enumerate(paths):
            vals = [f[m][1] for f in flat]
            v0 = vals[0]
            if isinstance(v0, np.ndarray):
                if id(v0) in first:                                        # the same array under two names
                    self.aliases[path] = first[id(v0)]
                    continue
                first[id(v0)] = path
            if isinstance(v0, np.ndarray) and v0.dtype.kind == "f":
                if any(v.shape != v0.shape for v in vals):
                    raise ValueError(f"{path}: array shapes differ between nodes")
                if path[-1] == "psi":                                    # eigenvector signs are arbitrary
                    vals = [_sign_fixed(v) for v in vals]
                if all(np.array_equal(v, v0) for v in vals[1:]):
                    self.constants[path] = v0
                    continue
                columns.append(np.stack([np.ravel(v) for v in vals]))
                self.slots.append((path, offset, v0.shape))
                offset += v0.size
            elif isinstance(v0, (float, np.floating)):
                if all(v == v0 for v in vals[1:]):
                    self.constants[path] = v0
                    continue
                columns.append(np.array(vals, float)[:, None])
                self.slots.append((path, offset, None))
                offset += 1
            else:
                if any(not _same(v, v0) for v in vals[1:]):
                    raise ValueError(f"{path}: non-numeric entries differ between nodes")
                self.constants[path] = v0
        self.data = np.concatenate(columns, axis=1) if columns else np.zeros((len(levels), 0))
        self.nbytes = self.data.nbytes

    def weights(self, a_s, a_r):
        """(n_sites, n_nodes) interpolation weights for the arm amplitudes."""
        ws = lagrange_weights(self.nodes_s, a_s)
        wr = lagrange_weights(self.nodes_r, a_r)
        return (ws[:, :, None] * wr[:, None, :]).reshape(ws.shape[0], -1)

    def at(self, w):
        """Interpolated levels for one row of ``weights``."""
        return self.unpack(w @ self.data)

    def unpack(self, vec):
        """Levels from one row of ``weights @ data`` (rows are done in blocks, one matrix product)."""
        out = {}
        for path, value in self.constants.items():
            _set(out, path, value)
        for path, offset, shape in self.slots:
            if shape is None:
                _set(out, path, float(vec[offset]))
                continue
            arr = vec[offset:offset + int(np.prod(shape))].reshape(shape)
            if path[-1] == "psi":                     # axis_levels normalizes sum |psi|^2 = 1 on the grid
                arr = arr / np.sqrt(np.sum(arr**2, axis=1))[:, None]
            _set(out, path, arr)
        for path, target in self.aliases.items():
            node = out
            for k in target:
                node = node[k]
            _set(out, path, node)
        return out


PSI_LOBE_FRACTION = 0.1          # sign convention: the first lobe above this fraction of max |psi| is positive


def _sign_fixed(psi):
    """Wavefunctions (rows) with the sign that makes the outermost lobe on the
    -x side positive: a convention that follows each level continuously as
    the well changes (an overlap with a reference node does not, for high
    levels over a wide range of trap frequencies)."""
    psi = np.asarray(psi, float)
    big = np.abs(psi) >= PSI_LOBE_FRACTION * np.max(np.abs(psi), axis=1, keepdims=True)
    first = np.argmax(big, axis=1)
    sign = np.where(psi[np.arange(psi.shape[0]), first] < 0, -1.0, 1.0)
    return psi * sign[:, None]


SLIM_DROP = ("psi", "x", "v_hz")  # per-state arrays dropped from the stored site spectra (~0.1 MB per site)


def slim_levels(lv):
    """``axis_levels`` output without the wavefunctions, position grid and
    potentials: the energies, well positions and purities stay (other
    extensions read levels[state]['energies_hz'])."""
    out = {}
    for state, d in lv.items():
        if not isinstance(d, dict):
            out[state] = d
            continue
        e = {k: v for k, v in d.items() if k not in SLIM_DROP and k != "sites"}
        if isinstance(d.get("sites"), dict):
            e["sites"] = {k: {kk: vv for kk, vv in w.items() if kk not in SLIM_DROP} for k, w in d["sites"].items()}
        out[state] = e
    return out


def _same(a, b):
    try:
        return bool(np.all(a == b))
    except Exception:
        return a is b


def _node_key(cfg, b, states, n_levels, grid_points, half_width_m):
    frame = None if cfg.frame is None else tuple(np.round(np.asarray(cfg.frame, float).ravel(), 15))
    fields = tuple((k, v) for k, v in vars(cfg).items() if k != "frame")
    return (fields, frame, tuple(np.asarray(b, float)), tuple(states), n_levels, grid_points, half_width_m)


def _lv_bytes(lv):
    return sum(v.nbytes for _, v in _leaves(lv) if isinstance(v, np.ndarray))


class NodeCache(dict):
    """Node level calculations, the oldest dropped beyond NODE_CACHE_BYTES."""

    def __init__(self, max_bytes=NODE_CACHE_BYTES):
        super().__init__()
        self.max_bytes, self.bytes = max_bytes, 0

    def __setitem__(self, key, lv):
        if key in self:
            self.bytes -= _lv_bytes(dict.__getitem__(self, key))
            del self[key]
        super().__setitem__(key, lv)
        self.bytes += _lv_bytes(lv)
        while self.bytes > self.max_bytes and len(self) > 1:
            old = next(iter(self))
            self.bytes -= _lv_bytes(dict.__getitem__(self, old))
            dict.__delitem__(self, old)


# -------------------------------------------------------------- the grid
@dataclass
class GridResult:
    index: np.ndarray                  # site indices along each axis (-half .. half)
    spacing_m: dict                    # lattice period along lab x, y, z
    positions_m: np.ndarray            # (n, n, n, 3)
    amplitudes: dict                   # axis -> (A_static, A_rotating), each (n, n, n)
    spectra: list                      # n**3 site spectra, C order over [i, j, k]
    zeeman_hz: float
    temperature_k: float
    n_levels: int
    seconds: float = 0.0
    nodes_total: int = 0
    nodes_computed: int = 0
    direct_pairs: tuple = field(default_factory=tuple)
    misses: int = 0                    # pair levels spectrum() computed itself (0: all interpolated)
    quantity_error: str = ""           # the last quantity() call's unexpected failure, '' if none

    @property
    def shape(self):
        return (self.index.size,) * 3

    def site(self, i, j, k):
        """Spectrum at site indices (i, j, k), each in -half .. half."""
        h = self.index.size // 2
        return self.spectra[int(np.ravel_multi_index((i + h, j + h, k + h), self.shape))]

    def centre(self):
        return self.site(0, 0, 0)

    def quantity(self, fn):
        """fn(sp) -> float at every site, (n, n, n); NaN where the site was not
        computed or fn is undefined there (QUANTITY_EXPECTED_ERRORS). Any other
        exception also gives NaN, is recorded in ``quantity_error`` ('' when
        none) and warned about once per call (a typo in a quantity used to give
        an all-NaN map with no message)."""
        out = np.full(len(self.spectra), np.nan)
        n_bad, first = site_values(self.spectra, fn, out, range(len(self.spectra)))
        self.quantity_error = f"failed at {n_bad} sites: {first}" if n_bad else ""
        if n_bad:
            warnings.warn(f"grid quantity {self.quantity_error}", stacklevel=2)
        return out.reshape(self.shape)


class GridJob:
    """The grid calculation in steps (for a responsive window): ``step``
    computes level calculations for about ``budget_s`` and returns True when
    none are left; ``finish`` builds the site spectra (also in steps when
    called repeatedly through ``step``; ``run`` does it all)."""

    def __init__(self, site: SiteConfig, setups, cache=None, half=GRID_HALF, node_count=NODE_COUNT,
                 grid_points=NODE_GRID_POINTS, max_dn=1, states=(UPPER, LOWER), sites=None):
        t0 = time.perf_counter()
        self.site = replace(site, grid_points=grid_points)
        self.setups, self.max_dn, self.states = setups, max_dn, states
        self.cache = NodeCache() if cache is None else cache
        self.b = np.asarray(site.b_lab_gauss, float)
        self.spacing = {ax: setups[ax].spacing_m() for ax in PAIRS}
        self.index = np.arange(-half, half + 1)
        self.positions = grid_positions(self.spacing, half)
        self.n_sites = self.positions[..., 0].size
        self.amps, self.nodes, self.templates = {}, {}, {}
        self.node_count = node_count
        for ax in PAIRS:
            a_s, a_r = setups[ax].amplitudes(self.positions)
            self.amps[ax] = (a_s.ravel(), a_r.ravel())
            self.nodes[ax] = (chebyshev_nodes(a_s.min(), a_s.max(), node_count),
                              chebyshev_nodes(a_r.min(), a_r.max(), node_count))
            self.templates[ax] = setups[ax].config_at(np.zeros(3))
        if self.site.n_levels is None:
            # from the temperature (sidebands.n_levels_for), with the pairs' trap frequencies at the centre
            self.site = replace(self.site, n_levels=sb.n_levels_for(
                site.temperature_k, [self.templates[ax].trap_frequency_hz for ax in PAIRS]))
        self.direct = set()
        self.pending = [(ax, vs, vr) for ax in PAIRS for vs in self.nodes[ax][0] for vr in self.nodes[ax][1]
                        if self._key(ax, vs, vr) not in self.cache]
        self.nodes_total = sum(n[0].size * n[1].size for n in self.nodes.values())
        self.nodes_computed = 0
        self._todo0 = len(self.pending)
        # sites: (i, j, k) indices to compute (default all); the node ranges still span the whole block
        n = self.index.size
        self.order = list(range(self.n_sites)) if sites is None else sorted(
            {int(np.ravel_multi_index(tuple(int(v) + half for v in ijk), (n, n, n))) for ijk in sites})
        self.tables, self.spectra, self.misses, self._next = None, [None] * self.n_sites, 0, 0
        self._seconds = time.perf_counter() - t0

    def _cfg(self, ax, vs, vr):
        return replace(self.templates[ax], static_amplitude=float(vs), rotating_amplitude=float(vr))

    def _key(self, ax, vs, vr):
        return _node_key(self._cfg(ax, vs, vr), self.b, self.states, self.site.n_levels, self.site.grid_points,
                         self.site.grid_half_width_m)

    def _levels(self, cfg):
        return sb.axis_levels(cfg, self.b, self.site.n_levels, self.site.grid_points,
                              self.site.grid_half_width_m, self.states)

    @property
    def progress(self):
        """(steps done, steps in total): level calculations, then site spectra in blocks of 81."""
        todo = len(self.pending)
        blocks = int(np.ceil(len(self.order) / SITE_BLOCK))
        built = self._next // SITE_BLOCK
        return self._todo0 - todo + built, self._todo0 + blocks

    @property
    def done(self):
        return not self.pending and self._next == len(self.order)

    def step(self, budget_s=0.4, cancelled=lambda: False):
        """Work for about budget_s; True when the grid is complete."""
        t0 = time.perf_counter()
        while self.pending and not cancelled():
            ax, vs, vr = self.pending.pop(0)
            key = self._key(ax, vs, vr)
            if key not in self.cache:
                self.cache[key] = self._levels(self._cfg(ax, vs, vr))
                self.nodes_computed += 1
            if time.perf_counter() - t0 > budget_s:
                self._seconds += time.perf_counter() - t0
                return False
        if self.pending or cancelled():
            self._seconds += time.perf_counter() - t0
            return False
        if self.tables is None:
            self._build_tables()
        while self._next < len(self.order) and not cancelled():
            self._site_spectrum(self._next)
            self._next += 1
            if time.perf_counter() - t0 > budget_s:
                break
        self._seconds += time.perf_counter() - t0
        return self.done

    def _build_tables(self):
        self.tables, self.weights = {}, {}
        for ax in PAIRS:
            if ax in self.direct:
                continue
            nodes_s, nodes_r = self.nodes[ax]
            levels = [self.cache[self._key(ax, vs, vr)] for vs in nodes_s for vr in nodes_r]
            try:
                self.tables[ax] = LevelTable(nodes_s, nodes_r, levels)
            except ValueError:
                self.direct.add(ax)
                continue
            self.weights[ax] = self.tables[ax].weights(*self.amps[ax])
        self._block = (0, {ax: np.zeros((0, 0)) for ax in self.tables})
        self._direct_cache = NodeCache(max_bytes=self.cache.max_bytes if hasattr(self.cache, "max_bytes")
                                       else NODE_CACHE_BYTES)

    def site_config(self, m):
        """The exact pair configurations at site m (flat index)."""
        return [replace(self.templates[ax], static_amplitude=float(self.amps[ax][0][m]),
                        rotating_amplitude=float(self.amps[ax][1][m])) for ax in PAIRS]

    def _site_spectrum(self, pos):
        m = self.order[pos]
        axes = self.site_config(m)
        site = replace(self.site, axes=axes)
        given = {}
        for cfg in axes:
            ax = cfg.axis
            key = sb._levels_key(cfg, self.b, site, self.states)
            if ax in self.direct:
                k2 = self._key(ax, cfg.static_amplitude, cfg.rotating_amplitude)
                lv = self._direct_cache.get(k2)
                if lv is None:
                    lv = self._levels(cfg)
                    self._direct_cache[k2] = lv
                    self.nodes_computed += 1
            else:
                start, vecs = self._block
                if not start <= pos < start + vecs[ax].shape[0]:
                    rows = self.order[pos:pos + SITE_BLOCK]
                    self._block = start, vecs = pos, {a: self.weights[a][rows] @ t.data
                                                      for a, t in self.tables.items()}
                lv = self.tables[ax].unpack(vecs[ax][pos - start])
            given[key] = lv
        n_given = len(given)
        sp = sb.spectrum(site, self.states, self.max_dn, cache=given)
        self.misses += len(given) - n_given          # levels spectrum() had to compute itself (expected 0)
        for d in sp["axes"].values():
            if "levels" in d:
                d["levels"] = slim_levels(d["levels"])
        self.spectra[m] = sp

    def finish(self):
        if not self.done:
            raise RuntimeError("the grid is not complete; call step() until it returns True")
        return GridResult(index=self.index, spacing_m=self.spacing, positions_m=self.positions,
                          amplitudes={ax: tuple(a.reshape((self.index.size,) * 3) for a in self.amps[ax])
                                      for ax in PAIRS},
                          spectra=self.spectra, zeeman_hz=float(self.spectra[self.order[0]]["zeeman_hz"]),
                          temperature_k=self.site.temperature_k, n_levels=self.site.n_levels,
                          seconds=self._seconds, nodes_total=self.nodes_total,
                          nodes_computed=self.nodes_computed, direct_pairs=tuple(sorted(self.direct)),
                          misses=self.misses)

    def run(self, progress=None, cancelled=lambda: False):
        """The whole calculation; None if cancelled."""
        while not self.step(cancelled=cancelled):
            if cancelled():
                return None
            if progress is not None:
                progress(*self.progress)
        return self.finish()


def compute_grid(site, setups, cache=None, **kw):
    """GridResult of the site configuration (temperature, field, level count;
    ``site.n_levels`` None: from the temperature, ``sidebands.n_levels_for``)
    with every pair built from ``setups[axis]`` at every site."""
    return GridJob(site, setups, cache, **kw).run()


def direct_site(site: SiteConfig, setups, r_lab, max_dn=1, grid_points=NODE_GRID_POINTS, cache=None):
    """Reference for one site: the exact pair configurations at r_lab and
    ``sidebands.spectrum`` (no interpolation)."""
    axes = [setups[cfg.axis].config_at(r_lab) for cfg in site.axes]
    return sb.spectrum(replace(site, axes=axes, grid_points=grid_points), max_dn=max_dn, cache=cache)


# ------------------------------------------------------------ quantities
def sideband_ratio(sp, axis, dn=1):
    """Sideband-to-carrier ratio of one pair: its own dn and dn = 0 weights
    ('dn_weights', so another pair's vanishing carrier does not matter);
    NaN when the pair's carrier weight is 0 (wells far apart at mid angles)."""
    a = sp["axes"][axis]
    if "dn_weights" in a:
        w0 = a["dn_weights"].get(0, 0.0)
        return a["dn_weights"].get(dn, 0.0) / w0 if w0 > 0 else float("nan")
    lines = collapse_lines(sp["lines"], axis)
    w = sp["carrier_weight"]
    return lines.get((axis, dn), (0.0, 0.0))[1] / w if w > 0 else float("nan")


def builtin_quantities():
    """name -> fn(sp) -> float; the names carry the units."""
    q = {"carrier shift from the free-atom line (Hz)": lambda sp: sp["carrier_shift_hz"]}
    for ax in PAIRS:
        q[f"{ax} pair: trap frequency of |3,−3⟩ (kHz)"] = (lambda sp, ax=ax: sp["axes"][ax]["f_lower_hz"] / 1e3)
    for ax in PAIRS:
        q[f"{ax} pair: sideband / carrier, Δn = +1"] = (lambda sp, ax=ax: sideband_ratio(sp, ax, 1))
    for ax in PAIRS:
        # to the nearest |3,-3> well: past theta ~ pi/2 that is a neighbouring site's (displacement_m is the
        # own-site well's, which the atom no longer overlaps)
        q[f"{ax} pair: well separation x(|3,−3⟩) − x(|4,−4⟩), nearest well (nm)"] = (
            lambda sp, ax=ax: sp["axes"][ax].get("nearest_displacement_m", sp["axes"][ax]["displacement_m"]) * 1e9)
    return q


# ------------------------------------------------------- fits and sums
def fit_plane(values, positions):
    """Least-squares fit V = c + g . (r - r_mean) over the sites (NaNs
    dropped), as readstats' planefit3d does on its site maps (unweighted
    here). Returns dict(offset, gradient (3,) per metre, residual_rms)."""
    v = np.asarray(values, float).ravel()
    r = np.asarray(positions, float).reshape(-1, 3)
    ok = np.isfinite(v)
    if ok.sum() < 4:
        return dict(offset=float("nan"), gradient=np.full(3, np.nan), residual_rms=float("nan"))
    centre = r[ok].mean(axis=0)
    a = np.column_stack([np.ones(ok.sum()), r[ok] - centre])
    coef, *_ = np.linalg.lstsq(a, v[ok], rcond=None)
    resid = v[ok] - a @ coef
    return dict(offset=float(coef[0]), gradient=coef[1:].copy(), residual_rms=float(np.sqrt(np.mean(resid**2))))


def slice_means(values, axis):
    """Mean of a (n, n, n) site quantity over the other two axes: one value
    per slice index along ``axis`` (0, 1, 2 = x, y, z)."""
    others = tuple(a for a in range(3) if a != axis)
    return np.nanmean(np.asarray(values, float), axis=others)


def line_fit(index, values):
    """Straight line through per-slice values: (slope per slice, intercept)."""
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.asarray(index, float)[ok], np.asarray(values, float)[ok], 1)
    return float(slope), float(intercept)


def parabola_vertex(index, values):
    """Vertex (slice index) and curvature of a parabola through per-slice values
    (readstats fits the sideband spacing this way: the vertex is the slice the
    beam is centred on)."""
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    c2, c1, c0 = np.polyfit(np.asarray(index, float)[ok], np.asarray(values, float)[ok], 2)
    return (float(-c1 / (2 * c2)) if c2 != 0 else float("nan")), float(c2)


def _sticks(spectra):
    pos, wt = [], []
    for sp in spectra:
        if sp is None:
            continue
        c = sp["carrier_shift_hz"]
        pos.append(c)
        wt.append(sp["carrier_weight"])
        for _, _, off, w in sp["lines"]:
            pos.append(c + off)
            wt.append(w)
    return np.asarray(pos), np.asarray(wt)


def ensemble_curve_direct(spectra, f_hz, width_hz):
    """Mean line strength of the sites on a detuning axis from the free-atom
    line (Lorentzians summed directly; the reference for ``ensemble_curve``)."""
    spectra = [sp for sp in spectra if sp is not None]
    f = np.asarray(f_hz, float)
    total = np.zeros_like(f)
    for sp in spectra:
        total += line_shape(f - sp["carrier_shift_hz"], sp["lines"], sp["carrier_weight"], width_hz)
    return total / len(spectra)


def ensemble_curve(spectra, f_hz, width_hz):
    """``ensemble_curve_direct`` on a uniform axis by binning the lines onto
    the axis (linear sharing between the two nearest points) and one FFT
    convolution with the Lorentzian. Lines beyond the axis by more than its
    own span are dropped."""
    spectra = [sp for sp in spectra if sp is not None]
    f = np.asarray(f_hz, float)
    df = f[1] - f[0]
    n = f.size
    pad = n
    g0 = f[0] - pad * df
    pos, wt = _sticks(spectra)
    u = (pos - g0) / df
    keep = (u >= 0) & (u < n + 2 * pad - 1)
    i0 = np.floor(u[keep]).astype(int)
    frac = u[keep] - i0
    hist = np.zeros(n + 2 * pad)
    np.add.at(hist, i0, wt[keep] * (1 - frac))
    np.add.at(hist, i0 + 1, wt[keep] * frac)
    m = hist.size
    lags = np.arange(-(m - 1), m) * df
    kernel = width_hz**2 / (lags**2 + width_hz**2)
    full = fftconvolve(hist, kernel, mode="full")       # index t: sum_s hist[s] kernel[t - s + m - 1]
    out = full[m - 1 + pad: m - 1 + pad + n]
    return out / len(spectra)


def slice_curves(result: GridResult, axis, f_hz, width_hz):
    """Mean spectrum of every slice along ``axis`` (0, 1, 2): (n, len(f))."""
    n = result.index.size
    idx = np.arange(len(result.spectra)).reshape(result.shape)
    out = []
    for s in range(n):
        members = np.take(idx, s, axis=axis).ravel()
        out.append(ensemble_curve([result.spectra[m] for m in members], f_hz, width_hz))
    return np.array(out)
