"""The spectrum across the 9 x 9 x 9 block of sites (lattice_sim.spectra.grid)
and the explorer extension that shows it (scripts/explorer_grid.py)."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lattice_sim.constants import LATTICE_WAVELENGTH_M
from lattice_sim.spectra import grid as G
from lattice_sim.spectra.beams import NOMINAL_HALF_ANGLE, gradients

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


se = _load("sideband_explorer")
sys.modules.setdefault("sideband_explorer", se)
eg = _load("explorer_grid")

UM = 1e-6
CACHE = {}                       # node level calculations shared by the tests below


def settings_with(**pair_x):
    s = se.default_settings()
    s["pairs"]["x"].update(pair_x)
    return s


Z_LINE = [(0, 0, k) for k in range(-4, 5)]
CORNERS = [(i, j, k) for i in (-4, 0, 4) for j in (-4, 0, 4) for k in (-4, 0, 4)]


def grid_of(settings, waist_m=None, cache=CACHE, sites=None):
    """The block for explorer settings (sites: only these (i, j, k), the rest None)."""
    site, setups = eg.setups_from_settings(se, settings)
    if waist_m is not None:
        for ax, st in setups.items():
            for arm in (st.geometry.static, st.geometry.rotating):
                arm.waist_m = waist_m
    return G.compute_grid(site, setups, cache=cache, sites=sites), site, setups


@pytest.fixture(scope="module")
def nominal():
    return grid_of(se.default_settings())


@pytest.fixture(scope="module")
def plane_waves():
    """Infinite waist, every arm centred: one node per pair."""
    return grid_of(se.default_settings(), waist_m=np.inf, cache={}, sites=CORNERS)


@pytest.fixture(scope="module")
def single_arm():
    """The EOM arm of the x pair 10 um off along its static polarization (lab z)."""
    return grid_of(settings_with(rotating_um=10.0))


def carrier(res):
    return res.quantity(lambda sp: sp["carrier_shift_hz"])


def f_x(res):
    return res.quantity(lambda sp: sp["axes"]["x"]["f_lower_hz"])


def z_line_slope(res, values):
    """Hz per um of a straight line through the sites on the z axis (x = y = 0)."""
    h = res.index.size // 2
    return np.polyfit(res.positions_m[h, h, :, 2], values[h, h, :], 1)[0] * UM


# --------------------------------------------------------------- geometry
def test_rayleigh_range_makes_the_waist_constant_over_the_block():
    assert G.rayleigh_waist_change(20 * UM) < 2e-6           # z_R = 13 mm for w = 60 um


def test_site_positions_use_each_pairs_period():
    s = se.default_settings()
    s["pairs"]["y"]["dalpha_deg"] = 0.2
    _, setups = eg.setups_from_settings(se, s)
    sp = {ax: st.spacing_m() for ax, st in setups.items()}
    assert sp["x"] == pytest.approx(LATTICE_WAVELENGTH_M / (2 * np.sin(NOMINAL_HALF_ANGLE)), rel=1e-12)
    assert sp["z"] == pytest.approx(sp["x"], rel=1e-12)
    assert sp["y"] == pytest.approx(LATTICE_WAVELENGTH_M / (2 * np.sin(NOMINAL_HALF_ANGLE + np.deg2rad(0.2))),
                                    rel=1e-12)
    pos = G.grid_positions(sp)
    assert pos.shape == (9, 9, 9, 3)
    assert pos[8, 0, 4] == pytest.approx([4 * sp["x"], -4 * sp["y"], 0.0])


def test_centre_configs_are_the_explorers_site():
    """The grid's centre site is the explorer's single site, for offsets along
    either direction and a crossing-angle change."""
    for direction in ("pol", "fringe"):
        s = settings_with(static_um=7.0, rotating_um=-12.0, dalpha_deg=0.15)
        s["pairs"]["z"].update(rotating_um=5.0, depth=0.8)
        s["offset_dir"] = direction
        site, setups = eg.setups_from_settings(se, s)
        for cfg in site.axes:
            c = setups[cfg.axis].config_at(np.zeros(3))
            for field in ("trap_frequency_hz", "spacing_m", "theta", "static_amplitude", "rotating_amplitude"):
                assert getattr(c, field) == pytest.approx(getattr(cfg, field), rel=1e-12), (cfg.axis, field)
            assert np.allclose(c.b_local(site.b_lab_gauss), cfg.b_local(site.b_lab_gauss), atol=1e-12)


def test_each_pair_sees_its_offsets_along_its_own_frame():
    """Offsets 'along the static polarization' are lab z for the x and y
    pairs and lab x for the z pair; 'along the fringes' are lab x, y, z."""
    probe = {"x": np.array([[0, 0, 1.0], [1.0, 0, 0]]), "y": np.array([[0, 0, 1.0], [0, 1.0, 0]]),
             "z": np.array([[1.0, 0, 0], [0, 0, 1.0]])}         # rows: pol direction, fringe direction
    for direction, row in (("pol", 0), ("fringe", 1)):
        s = se.default_settings()
        for ax in "xyz":
            s["pairs"][ax].update(static_um=10.0, rotating_um=10.0)
        s["offset_dir"] = direction
        _, setups = eg.setups_from_settings(se, s)
        for ax, st in setups.items():
            u = probe[ax][row] * 10 * UM
            a_toward, _ = st.amplitudes(u[None, :])
            a_away, _ = st.amplitudes(-u[None, :])
            assert a_toward[0] == pytest.approx(1.0, abs=1e-3), (ax, direction)   # site on the arm's axis
            assert a_away[0] < 0.9


# ----------------------------------------------------------- the methods
def test_interpolated_levels_reach_spectrum_unchanged(plane_waves):
    """With one node per pair (plane waves: every site the same) the levels
    handed to sidebands.spectrum are the node's own, so a site's spectrum is
    the direct calculation exactly, and spectrum() computed nothing itself."""
    res, site, setups = plane_waves
    assert res.misses == 0 and res.direct_pairs == ()
    d = G.direct_site(site, setups, res.positions_m[0, 0, 0])
    g = res.site(-4, -4, -4)
    assert g["carrier_shift_hz"] == pytest.approx(d["carrier_shift_hz"], rel=1e-12)
    assert g["carrier_weight"] == pytest.approx(d["carrier_weight"], rel=1e-12)
    assert len(g["lines"]) == len(d["lines"])
    for a, b in zip(g["lines"], d["lines"]):
        assert a[:2] == b[:2] and a[2] == pytest.approx(b[2], abs=1e-9)
        assert a[3] == pytest.approx(b[3], rel=1e-9, abs=1e-20)
    for state in ((4, -4), (3, -3)):                      # levels kept without wavefunctions (memory)
        kept = g["axes"]["x"]["levels"][state]
        assert "psi" not in kept and "v_hz" not in kept
        assert np.allclose(kept["energies_hz"], d["axes"]["x"]["levels"][state]["energies_hz"], rtol=0, atol=1e-6)


def test_sideband_ratio_guards_a_vanishing_carrier():
    sp = {"axes": {"x": {"dn_weights": {-1: 0.1, 0: 0.0, 1: 0.2}}}, "carrier_weight": 0.0, "lines": []}
    assert np.isnan(G.sideband_ratio(sp, "x"))
    sp["axes"]["x"]["dn_weights"][0] = 0.5
    assert G.sideband_ratio(sp, "x") == pytest.approx(0.4)
    assert np.isnan(G.line_fit([0, 1, 2], [np.nan] * 3)[0])


def test_settings_key_and_the_centred_reference():
    s = se.default_settings()
    other = se.copy_settings(s)
    other["offset_dir"] = "fringe"
    other["glob"]["width_hz"] = 900.0                     # display only
    assert eg.grid_settings_key(se, s) == eg.grid_settings_key(se, other)
    moved = settings_with(rotating_um=10.0, depth=0.9)
    centred = eg.centred_settings(se, moved)
    assert centred["pairs"]["x"]["rotating_um"] == 0.0 and centred["pairs"]["x"]["depth"] == 0.9
    back = eg.centred_settings(se, settings_with(rotating_um=10.0))
    assert eg.grid_settings_key(se, back) == eg.grid_settings_key(se, s)


def test_chebyshev_interpolation_is_exact_for_polynomials():
    nodes = G.chebyshev_nodes(0.4, 0.9, 5)
    x = np.linspace(0.3, 1.0, 11)
    w = G.lagrange_weights(nodes, x)
    p = lambda t: 1 - 2 * t + 3 * t**3 - t**4
    assert np.allclose(w @ p(nodes), p(x), atol=1e-12)
    assert G.chebyshev_nodes(0.7, 0.7, 5).size == 1


def test_node_grid_points_match_the_site_grid():
    """601 instead of 1201 position points: 0.03 Hz on f, 1e-4 Hz on the carrier."""
    s = settings_with(rotating_um=40.0)
    site, setups = eg.setups_from_settings(se, s)
    r = np.array([4, 4, 4]) * 4.8374 * UM
    a = G.direct_site(site, setups, r, grid_points=G.NODE_GRID_POINTS)
    b = G.direct_site(site, setups, r, grid_points=site.grid_points)
    assert abs(a["carrier_shift_hz"] - b["carrier_shift_hz"]) < 1e-3
    for ax in "xyz":
        assert abs(a["axes"][ax]["f_lower_hz"] - b["axes"][ax]["f_lower_hz"]) < 0.1
        assert G.sideband_ratio(a, ax) == pytest.approx(G.sideband_ratio(b, ax), rel=1e-4)


def test_interpolation_matches_direct_evaluation():
    """The largest offset the explorer allows (40 um, one arm): every pair
    interpolated over its arm amplitudes against the direct calculation at
    the centre, the corners and an edge site."""
    s = settings_with(rotating_um=40.0)
    picks = [(0, 0, 0), (-4, -4, -4), (4, -4, 2)]
    res, site, setups = grid_of(s, sites=picks)
    assert res.misses == 0 and res.direct_pairs == ()
    for ijk in picks:
        idx = tuple(i + 4 for i in ijk)
        d = G.direct_site(site, setups, res.positions_m[idx])
        g = res.site(*ijk)
        assert abs(g["carrier_shift_hz"] - d["carrier_shift_hz"]) < 0.01
        assert abs(g["carrier_weight"] / d["carrier_weight"] - 1) < 1e-4   # 3e-5 at the corners
        for ax in "xyz":
            assert abs(g["axes"][ax]["f_lower_hz"] - d["axes"][ax]["f_lower_hz"]) < 2.0
            assert abs(G.sideband_ratio(g, ax) - G.sideband_ratio(d, ax)) < 1e-4


# ------------------------------------------------------------ the physics
def test_plane_waves_give_identical_sites(plane_waves):
    """Infinite waist and centred arms: every site is the explorer's site."""
    res = plane_waves[0]
    assert res.nodes_total == 3
    c, f = carrier(res), f_x(res)
    assert np.nanmax(c) == np.nanmin(c) and np.nanmax(f) == np.nanmin(f)
    ref = se.compute(se.default_settings())["sp"]
    assert c[4, 4, 4] == pytest.approx(ref["carrier_shift_hz"], abs=1e-3)
    assert f[4, 4, 4] == pytest.approx(ref["axes"]["x"]["f_lower_hz"], abs=0.1)
    centre = res.centre()
    width = 300.0
    freq = np.linspace(-25e3, 25e3, 2001)
    assert np.allclose(G.ensemble_curve(res.spectra, freq, width), se.spectrum_curve(centre, freq, width),
                       rtol=2e-3, atol=1e-7)


def test_nominal_block_is_symmetric_with_no_gradient(nominal):
    """With every arm centred the beams' envelope makes the sites differ, but
    symmetrically: no linear gradient anywhere; f_x falls towards the edges."""
    res = nominal[0]
    c, f = carrier(res), f_x(res)
    assert np.allclose(c, c[::-1, ::-1, ::-1], rtol=0, atol=1e-6)   # inversion r -> -r
    assert np.allclose(f, f[::-1, ::-1, ::-1], rtol=0, atol=1e-6)
    for v in (c, f):
        assert np.all(np.abs(G.fit_plane(v, res.positions_m)["gradient"]) * UM < 1e-6)
    assert f[4, 4, 4] == f.max()
    assert 0.80 < f[0, 4, 0] / f[4, 4, 4] < 0.85                  # exp(-(x^2+z^2)/w^2), x = z = 19.35 um
    assert np.ptp(c) > 500                                         # the x-y saddle below


def test_crossing_arms_unbalance_off_axis_sites_through_the_ellipticity():
    """At x = 19 um the arms of the x pair (crossing at +/-5 deg) are 1.7 um
    nearer and farther at y = +/-19 um: their imbalance moves the carrier by
    hundreds of Hz at ER 3e4 and not at all for pure linear light."""
    out = {}
    for er in (3e4, np.inf):
        s = se.default_settings()
        for ax in "xyz":
            s["pairs"][ax]["er"] = er
        site, setups = eg.setups_from_settings(se, s)
        a = 4 * 4.8374 * UM
        out[er] = [G.direct_site(site, setups, np.array([a, y, 0.0]))["carrier_shift_hz"] for y in (a, 0.0, -a)]
    c = out[3e4]
    assert c[0] - c[1] < -250 and c[2] - c[1] > 350               # -1110, -797, -370 Hz
    lin = out[np.inf]
    assert lin[0] == pytest.approx(lin[2], abs=1e-3) and abs(lin[0] - lin[1]) < 100


def test_single_arm_offset_reproduces_the_findings_table(single_arm):
    """findings 2026-10-08: EOM arm +10 um in z, sites along z (x = y = 0):
    dcarrier/dz ~ +28 Hz/um, df_x/dz ~ +47 Hz/um (5 sites over +/-20 um there,
    9 sites over +/-19.35 um here; 10 % allowed for the different sampling)."""
    res = single_arm[0]
    dc, df = z_line_slope(res, carrier(res)), z_line_slope(res, f_x(res))
    assert dc == pytest.approx(28, rel=0.10)                      # 29.2
    assert df == pytest.approx(47, rel=0.10)                      # 48.2
    g = G.fit_plane(carrier(res), res.positions_m)["gradient"] * UM
    assert abs(g[0]) < 1e-6 and abs(g[1]) < 1e-6                  # nothing along x or y


def test_differential_offset_gives_a_carrier_gradient_without_a_sideband_gradient():
    res = grid_of(settings_with(static_um=10.0, rotating_um=-10.0), sites=Z_LINE)[0]
    assert abs(z_line_slope(res, f_x(res))) < 0.5
    assert z_line_slope(res, carrier(res)) == pytest.approx(-60, rel=0.10)   # findings: -60 Hz/um


def test_common_offset_gives_the_depth_gradient():
    """Both x arms +10 um along z. Method 1: the grid's f_x at z = +/-1 site.
    Method 2: the explorer's d ln f / d ln(depth) at the centre-site depth
    times the geometric depth gradient d ln(A_s A_r)/dz = 4 r0 / w^2.
    findings states df/f = 2 r0 / w^2 per unit length (harmonic, f ~ sqrt(depth))."""
    r0, w = 10 * UM, se.WAIST_M
    res = grid_of(settings_with(static_um=10.0, rotating_um=10.0), sites=[(0, 0, -1), (0, 0, 1)])[0]
    f = f_x(res)
    az = res.spacing_m["z"]
    dlnf_grid = (np.log(f[4, 4, 5]) - np.log(f[4, 4, 3])) / (2 * az)
    depth0 = np.exp(-2 * r0**2 / w**2)
    fs = []
    for d in (depth0 * 0.99, depth0 * 1.01):
        s = settings_with(depth=d)
        fs.append(se.compute(s)["sp"]["axes"]["x"]["f_lower_hz"])
    dlnf_dlnu = (np.log(fs[1]) - np.log(fs[0])) / (np.log(1.01) - np.log(0.99))
    assert dlnf_grid == pytest.approx(dlnf_dlnu * 4 * r0 / w**2, rel=0.01)
    assert dlnf_grid == pytest.approx(2 * r0 / w**2, rel=0.10)


def test_grid_gradients_agree_with_beams_gradients(single_arm):
    """Two routes to the same gradient: the grid sites at z = +/-2 sites
    (interpolated, all pairs from their geometry) against beams.gradients
    (direct, the x pair alone) at the same +/-9.67 um."""
    res, site, setups = single_arm
    az = res.spacing_m["z"]
    c, f = carrier(res), f_x(res)
    base = [setups["x"].base if cfg.axis == "x" else cfg for cfg in site.axes]
    g = gradients(replace(site, axes=base, grid_points=G.NODE_GRID_POINTS), "x", setups["x"].geometry,
                  half_span_m=2 * az)
    assert (c[4, 4, 6] - c[4, 4, 2]) / (4 * az) == pytest.approx(g["carrier_hz_per_m_pol"], rel=1e-4)
    assert (f[4, 4, 6] - f[4, 4, 2]) / (4 * az) == pytest.approx(g["f_lower_hz_per_m_pol"], rel=1e-4)
    assert abs(g["carrier_hz_per_m_fringe"]) * UM < 1e-6
    # the whole-block plane fit sees a smaller gradient than the centre line (edge sites are shallower)
    plane = G.fit_plane(c, res.positions_m)["gradient"][2] * UM
    assert 0.85 < plane / z_line_slope(res, c) < 0.97                 # 26.8 vs 29.2 Hz/um


def test_slice_line_is_the_plane_fit(single_arm):
    """readstats' slices (sums over the other two axes, straight line against
    the slice index) and the 3D plane fit give the same gradient on a full block."""
    res = single_arm[0]
    c = carrier(res)
    for axis in range(3):
        slope, _ = G.line_fit(res.index, G.slice_means(c, axis))
        a = res.spacing_m["xyz"[axis]]
        assert slope == pytest.approx(G.fit_plane(c, res.positions_m)["gradient"][axis] * a, abs=1e-9)
    vertex, curvature = G.parabola_vertex(res.index, G.slice_means(f_x(res), 2))
    assert curvature < 0 and 0.8 < vertex < 1.3        # beams centred 5 um (1.03 sites) above the centre


# -------------------------------------------------------------- sums
def test_binned_ensemble_matches_the_direct_sum(nominal):
    res = nominal[0]
    freq = np.linspace(-24e3, 24e3, 4001)
    some = res.spectra[::37]
    direct = G.ensemble_curve_direct(some, freq, 300.0)
    binned = G.ensemble_curve(some, freq, 300.0)
    assert np.max(np.abs(binned - direct) / direct.max()) < 1e-3


# ---------------------------------------------------------------- GUI
def test_drawing_functions_run_headless(single_arm):
    """Every panel draws, with tick labels and an x label on every axes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    res = single_arm[0]
    s = settings_with(rotating_um=10.0)
    values = carrier(res)

    def figure():
        fig = Figure(figsize=(11, 5.8), constrained_layout=True)
        FigureCanvasAgg(fig)
        return fig
    drawn = []
    for ref_mode, all_planes in ((eg.REFERENCES[2], True),):
        fig = figure()
        eg.draw_maps(fig, se, res, res, values, values, "carrier (Hz)", ref_mode, 1, 2, all_planes, s)
        drawn.append(fig)
    for follow in (eg.FOLLOW[1],):
        fig = figure()
        eg.draw_slices(fig, se, res, 300.0, follow, s)
        drawn.append(fig)
    fig = figure()
    eg.draw_array(fig, se, res, None, 300.0, values, values, "carrier (Hz)", eg.REFERENCES[1], s)
    drawn.append(fig)
    for fig in drawn:
        fig.canvas.draw()
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>" or not ax.get_visible():
                continue
            assert ax.get_xlabel(), ax.get_title()
            assert any(t.get_text() for t in ax.get_xticklabels())


def _tiny_result(spectra):
    n = 3
    return G.GridResult(index=np.arange(-1, 2), spacing_m={a: 4.8e-6 for a in G.PAIRS},
                        positions_m=np.zeros((n, n, n, 3)), amplitudes={}, spectra=spectra, zeeman_hz=0.0,
                        temperature_k=1e-6, n_levels=13)


def test_grid_quantity_reports_unexpected_errors():
    import warnings
    spectra = [dict(carrier_shift_hz=float(m), carrier_weight=0.0) for m in range(27)]
    spectra[5] = None                                              # a site not computed
    res = _tiny_result(spectra)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = res.quantity(lambda sp: sp["carrier_shift_hz"])
        assert np.isnan(v.ravel()[5]) and v.ravel()[6] == 6.0 and res.quantity_error == ""
        v = res.quantity(lambda sp: sp["missing"])                 # undefined: NaN, no message
        v = res.quantity(lambda sp: 1.0 / sp["carrier_weight"])
        assert np.all(np.isnan(v)) and res.quantity_error == ""
    with pytest.warns(UserWarning, match="AttributeError"):
        v = res.quantity(lambda sp: sp.carrier_shift_hz)            # a bug in the quantity: said
    assert np.all(np.isnan(v)) and res.quantity_error.startswith("failed at 26 sites: AttributeError")
    out = np.full(27, np.nan)
    assert G.site_values(spectra, lambda sp: sp.x, out, range(27))[0] == 26


def test_well_separation_quantity_uses_the_nearest_well():
    q = {k: f for k, f in G.builtin_quantities().items() if "well separation" in k}
    sp = dict(axes={ax: dict(displacement_m=-2.4e-6, nearest_displacement_m=0.1e-6) for ax in G.PAIRS})
    assert all("nearest" in k for k in q) and len(q) == 3
    assert [f(sp) for f in q.values()] == pytest.approx([100.0] * 3)


def test_node_grid_resolution_holds_at_60_levels():
    """NODE_GRID_POINTS (601, dx = 5 nm) against the 1201 of SiteConfig at the corner the explorer
    reaches: z pair at 0.3 depth, 60 levels. Level 59 moves by 64 Hz (1 % of its 5.7 kHz spacing),
    but both states move nearly together: carrier lines E_u(n) - E_l(n) change by < 0.6 Hz, the
    |dn| <= 2 sidebands by up to 5.8 Hz at the top (levels below 13: 0.33 Hz)."""
    from lattice_sim.spectra import sidebands as sb
    site = sb.nominal_site()
    z = replace([c for c in site.axes if c.axis == "z"][0], trap_frequency_hz=12.5e3 * np.sqrt(0.3))
    a = sb.axis_levels(z, site.b_lab_gauss, 60, grid_points=G.NODE_GRID_POINTS)
    b = sb.axis_levels(z, site.b_lab_gauss, 60, grid_points=1201)
    du = a[sb.UPPER]["energies_hz"] - b[sb.UPPER]["energies_hz"]
    dl = a[sb.LOWER]["energies_hz"] - b[sb.LOWER]["energies_hz"]
    assert np.max(np.abs(du)) < 100.0 and np.max(np.abs(dl)) < 100.0
    lines = du[None, :] - dl[:, None]                               # [n', n]
    near = np.abs(np.arange(60)[:, None] - np.arange(60)[None, :]) <= 2
    assert np.max(np.abs(np.diag(lines))) < 1.0
    assert np.max(np.abs(lines[near])) < 10.0
    assert np.max(np.abs(lines[:13, :13][near[:13, :13]])) < 0.5
    fa = sb.franck_condon(a[sb.UPPER]["psi"], a[sb.LOWER]["psi"])
    fb = sb.franck_condon(b[sb.UPPER]["psi"], b[sb.LOWER]["psi"])
    assert np.max(np.abs(fa - fb)) < 1e-3
