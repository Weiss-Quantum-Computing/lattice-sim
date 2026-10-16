"""The sideband model over the full polarization range theta in [-pi, pi]:
well tracking, neighbouring-site lines, adiabatic labelling, the exact
symmetries of the field, and the zero-point energy versus rotation."""

from dataclasses import replace

import numpy as np
import pytest

from lattice_sim.constants import H, HBAR, MASS_CS
from lattice_sim.spectra.sidebands import (LOWER, SITE_OFFSETS, UPPER, axis_levels, collapse_lines,
                                           energies_vs_angle, label_purity, nominal_site, spectrum,
                                           state_potentials, track_wells)

SITE = nominal_site()
B = SITE.b_lab_gauss
X_CFG = SITE.axes[0]
OTHER_STATES = [(3, -1), (4, 1), (4, -3), (3, 3), (4, 4)]


def with_x(site, **fields):
    return replace(site, axes=[replace(c, **fields) if c.axis == "x" else c for c in site.axes])


def lines_by_key(sp, axis):
    return {k: v for k, v in collapse_lines(sp["lines"], axis).items()}


# ------------------------------------------------------------------ labelling
@pytest.mark.parametrize("theta", [0.0, 0.03 * np.pi, 0.3 * np.pi, 0.7 * np.pi, np.pi])
def test_rank_labelling_equals_overlap_labelling_where_that_is_defined(theta):
    model = X_CFG.model(B)
    x = np.linspace(-X_CFG.spacing_m / 2, X_CFG.spacing_m / 2, 241)
    ours = state_potentials(model, (UPPER, LOWER), x, theta)
    ref = model.potentials((UPPER, LOWER), x, theta)
    for s in (UPPER, LOWER):
        assert np.max(np.abs(ours[s] - ref[s])) < 1e-4          # Hz, rounding of 9 GHz eigenvalues
    # other states: compare point by point wherever the overlap labelling is unambiguous
    energies, vectors = np.linalg.eigh(model.hamiltonians(x, theta))
    v_all = state_potentials(model, OTHER_STATES, x, theta)
    for s in OTHER_STATES:
        ov = np.abs(np.einsum("i,nij->nj", model._bias[s][1].conj(), vectors)) ** 2
        good = ov.max(axis=1) >= 0.5
        ref_s = energies[np.arange(x.size), ov.argmax(axis=1)] - model._bias[s][0]
        assert np.max(np.abs(v_all[s][good] - ref_s[good]), initial=0.0) < 1e-4


def test_spectrum_runs_where_the_overlap_labelling_raises():
    # 1.5 x depth, theta = 110 deg: |4,-4> is < 50 % of its bias-field state ~1 MHz up the well walls,
    # so StatePotential.potentials raises on the spectrum's grid; the adiabatic branch is used instead
    cfg = replace(X_CFG, trap_frequency_hz=19e3 * np.sqrt(1.5), theta=np.deg2rad(110))
    lv = axis_levels(cfg, B, 6)
    with pytest.raises(RuntimeError, match="strongly mixed"):
        cfg.model(B).potentials((UPPER, LOWER), lv[UPPER]["x"], cfg.theta)
    assert 0.8 < lv[UPPER]["purity"] < 0.9 and 0.9 < lv[LOWER]["purity"] < 0.95
    sp = spectrum(with_x(SITE, trap_frequency_hz=cfg.trap_frequency_hz, theta=cfg.theta), max_dn=1)
    assert np.isfinite(sp["carrier_shift_hz"]) and all(np.isfinite(l[3]) for l in sp["lines"])


def test_stretched_states_stay_mostly_pure_at_the_trap_minimum():
    model = X_CFG.model(B)
    for deg, low in [(0, 0.9999), (45, 0.99), (90, 0.6)]:
        th = np.deg2rad(deg)
        pos = track_wells(model, (UPPER, LOWER), [th])
        for s in (UPPER, LOWER):
            assert label_purity(model, s, pos[s], th)[0] > low
    # at theta = pi/2 the fictitious field tilts the local axis: 84 % |4,-4>, 62 % |3,-3>
    th = np.pi / 2
    pos = track_wells(model, (UPPER, LOWER), [th])
    assert label_purity(model, UPPER, pos[UPPER], th)[0] == pytest.approx(0.838, abs=0.005)
    assert label_purity(model, LOWER, pos[LOWER], th)[0] == pytest.approx(0.615, abs=0.005)


# ------------------------------------------------------------------ symmetries
@pytest.mark.parametrize("er, h", [(np.inf, -1), (1e2, -1), (1e2, 1)])
def test_field_symmetries_mirror_and_half_shift(er, h):
    """V(x; -theta) = V(-x; theta) and V(x; theta + pi) = V(x + a/2; theta), any state."""
    for cfg in nominal_site(extinction_ratio=er, handedness=h).axes:
        model = cfg.model(B)
        a = cfg.spacing_m
        x = np.linspace(-a, a, 33)
        states = [UPPER, LOWER] + OTHER_STATES
        for th in (0.2, 1.1, 2.6):
            v1, v2 = state_potentials(model, states, x, -th), state_potentials(model, states, -x, th)
            v3, v4 = state_potentials(model, states, x, th + np.pi), state_potentials(model, states, x + a / 2, th)
            for s in states:
                assert np.max(np.abs(v1[s] - v2[s])) < 1e-4
                assert np.max(np.abs(v3[s] - v4[s])) < 1e-4


@pytest.mark.parametrize("er, h", [(np.inf, -1), (1e2, -1), (1e2, 1)])
def test_each_well_moves_by_half_a_spacing_between_zero_and_pi(er, h):
    for cfg in nominal_site(extinction_ratio=er, handedness=h).axes:
        a = cfg.spacing_m
        pos = track_wells(cfg.model(B), (UPPER, LOWER), [np.pi, -np.pi])
        lv = axis_levels(replace(cfg, theta=np.pi), B, 3)
        for s in (UPPER, LOWER):
            assert abs(pos[s][0]) == pytest.approx(a / 2, abs=1e-9)
            assert pos[s][1] == pytest.approx(-pos[s][0], abs=1e-9)
            assert abs(lv[s]["x0_m"]) == pytest.approx(a / 2, abs=5e-11)
        # opposite directions: |4,-4> of site 0 meets |3,-3> of site +/-1
        assert np.sign(pos[UPPER][0]) == -np.sign(pos[LOWER][0])


# ------------------------------------------------------------------ tracking
def test_tracker_follows_state_potential_track():
    model = X_CFG.model(B)
    thetas = np.linspace(0, np.pi, 13)
    for s in (UPPER, LOWER):
        x_ref, f_ref = model.track(s, thetas)
        x = track_wells(model, (s,), thetas)[s]
        assert np.max(np.abs(x - x_ref)) < 2e-9
        e = energies_vs_angle(X_CFG, B, thetas, states=(s,))["states"][s]
        assert np.max(np.abs(e["x_m"] - x_ref)) < 0.2e-9
        assert np.max(np.abs(e["harmonic_hz"] / f_ref - 1)) < 1e-3


def test_well_positions_and_trap_frequencies_are_continuous_over_the_full_range():
    thetas = np.deg2rad(np.arange(-180, 181, 5.0))
    e = energies_vs_angle(X_CFG, B, thetas)
    a = X_CFG.spacing_m
    for s in (UPPER, LOWER):
        d = e["states"][s]
        # one minimum per period, moving monotonically away from the start; steepest next to 90 deg
        # (|3,-3> of the x pair: 0.6 um in the last 5 deg), far from the a/2 that would make it ambiguous
        assert np.max(np.abs(np.diff(d["x_m"]))) < a / 4
        half = d["x_m"][thetas >= 0]
        assert np.all(np.diff(half) * np.sign(half[-1]) > 0)
        # mirror symmetry of the field: even quantities even in theta, positions odd
        assert np.allclose(d["trap_hz"], d["trap_hz"][::-1], atol=1e-3)
        assert np.allclose(d["x_m"], -d["x_m"][::-1], atol=1e-11)
    # the two states move in opposite directions, by a/2 each at +/-180 deg
    assert e["states"][UPPER]["x_m"][-1] == pytest.approx(a / 2, abs=1e-11)
    assert e["states"][LOWER]["x_m"][-1] == pytest.approx(-a / 2, abs=1e-11)


# ------------------------------------------------------------------ spectrum over the range
@pytest.fixture(scope="module")
def cache():
    return {}


def x_spectrum(deg, cache, **kw):
    return spectrum(replace(with_x(SITE, theta=np.deg2rad(deg)), n_levels=13), max_dn=1, cache=cache, **kw)


def test_spectrum_is_defined_at_every_angle(cache):
    for deg in range(-180, 181, 30):
        sp = x_spectrum(deg, cache)
        x = sp["axes"]["x"]
        assert np.isfinite(sp["carrier_shift_hz"])
        assert 0.0 <= sp["carrier_weight"] <= 1.0
        assert all(np.isfinite(l[-1]) and l[-1] >= 0 for l in sp["lines"] + sp["site_lines"])
        assert 0.0 <= x["strength_in_lines"] <= 1.0 + 1e-9
        assert set(x["diagonal_sites"]) <= set(SITE_OFFSETS) and 0 in x["diagonal_sites"]
    # mid-range: the wells are a/2 apart, so the carrier and Delta n = +/-1 lines carry no strength,
    # and the strength sits far below the carrier (sum rule)
    sp = x_spectrum(90, cache)
    assert sp["carrier_weight"] == 0.0 and sp["axes"]["x"]["strength_in_lines"] == 0.0
    assert sp["axes"]["x"]["centroid_offset_hz"] < -500e3


@pytest.mark.parametrize("deg", [5.4, 30.0, 60.0])
def test_spectrum_is_even_in_theta_and_symmetric_about_90_degrees(deg, cache):
    sp = x_spectrum(deg, cache)
    for other, k_map in [(-deg, lambda k: -k), (180 - deg, lambda k: 1 - k), (deg - 180, lambda k: -1 - k)]:
        so = x_spectrum(other, cache)
        assert so["carrier_shift_hz"] == pytest.approx(sp["carrier_shift_hz"], abs=1e-3)
        assert so["carrier_weight"] == pytest.approx(sp["carrier_weight"], rel=1e-6, abs=1e-15)
        a, b = lines_by_key(sp, "x"), lines_by_key(so, "x")
        for key in a:
            assert b[key][1] == pytest.approx(a[key][1], rel=1e-6, abs=1e-15)
            assert b[key][0] == pytest.approx(a[key][0], abs=1e-3)
        # the same strength, carried by the mirrored / neighbouring final site
        for k, d in sp["axes"]["x"]["diagonal_sites"].items():
            assert so["axes"]["x"]["diagonal_sites"].get(k_map(k), 0.0) == pytest.approx(d, rel=1e-6, abs=1e-15)


def test_at_pi_the_neighbouring_site_carries_the_whole_carrier(cache):
    sp = x_spectrum(180, cache)
    x = sp["axes"]["x"]
    # the two wells coincide; their curvatures differ by the hyperfine scalar split (0.226 %)
    assert x["diagonal_sites"][1] > 1 - 1e-8 and x["diagonal_sites"][0] == 0.0
    assert sp["carrier_weight_same_site"] == 0.0
    assert x["nearest_site"] == 1 and abs(x["nearest_displacement_m"]) < 1e-11
    assert x["displacement_m"] == pytest.approx(-X_CFG.spacing_m, abs=1e-10)
    # per-site lines sum to the site-summed lines
    for key, (_, w) in lines_by_key(sp, "x").items():
        w_sites = sum(wt for ax, k, dn, off, wt in sp["site_lines"] if (ax, dn) == key)
        assert w_sites == pytest.approx(w, rel=1e-12, abs=1e-300)
    # periodic lattice: the k = +1 well has the k = 0 well's levels
    sites = x["levels"][LOWER]["sites"]
    assert np.max(np.abs(sites[1]["energies_hz"] - sites[0]["energies_hz"])) < 1e-3


def test_sum_rule_centroid_matches_the_explicit_lines():
    """Centroid and rms of the whole line-strength distribution: sum rule vs explicit Franck-Condon sum."""
    for deg in (5.4, 15.0):
        sp = spectrum(replace(with_x(SITE, theta=np.deg2rad(deg)), n_levels=30), max_dn=29)
        x = sp["axes"]["x"]
        lv = x["levels"]
        e_u, e_l = lv[UPPER]["energies_hz"], lv[LOWER]["energies_hz"]
        off = (e_u - e_u[0])[None, :] - (e_l - e_l[0])[:, None]          # [n', n]
        wt = x["thermal"][None, :] * sum(x["franck_condon_sites"].values())
        assert wt.sum() == pytest.approx(1.0, abs=1e-6)
        mean = np.sum(wt * off) / wt.sum()
        rms = np.sqrt(np.sum(wt * off**2) / wt.sum() - mean**2)
        assert x["centroid_offset_hz"] == pytest.approx(mean, abs=0.05)
        assert x["rms_width_hz"] == pytest.approx(rms, rel=1e-4)


# ------------------------------------------------------------------ zero-point energy
def test_zero_point_energy_of_the_sinusoidal_lattice():
    """theta = 0, pure linear light: V = V0 sin^2(q x) exactly, so E0 - V_min = f/2 - E_q/4 and
    E1 - E0 = f - E_q to first order in E_q = hbar^2 q^2 / 2m (16.04 Hz at a = 4.837 um)."""
    for cfg in SITE.axes:
        cfg = replace(cfg, theta=0.0, extinction_ratio=np.inf)
        e_q = H / (8 * MASS_CS * cfg.spacing_m**2)
        r = energies_vs_angle(cfg, B, [0.0])
        for s in (UPPER, LOWER):
            d = {k: v[0] for k, v in r["states"][s].items()}
            assert d["zpe_hz"] == pytest.approx(d["harmonic_hz"] / 2 - e_q / 4, abs=0.01)
            assert d["trap_hz"] == pytest.approx(d["harmonic_hz"] - e_q, abs=0.05)
            # the depth calibration refers to the centroid scalar shift; F = 4 / 3 differ by -0.226 %
            assert d["harmonic_hz"] / cfg.trap_frequency_hz == pytest.approx(1.0, abs=2e-3)


def _dvr_levels(model, state, theta, xc, half=0.8e-6, dx=5e-9):
    """Colbert-Miller sinc-DVR: an independent discretization of the same Hamiltonian."""
    x = xc + np.arange(-round(half / dx), round(half / dx) + 1) * dx
    v = state_potentials(model, (state,), x, theta)[state]
    d = np.arange(x.size)[:, None] - np.arange(x.size)[None, :]
    t = np.where(d == 0, np.pi**2 / 3, 2.0 * (-1.0) ** d / np.where(d == 0, 1, d) ** 2)
    t = t * HBAR**2 / (2 * MASS_CS * dx**2) / H
    return np.linalg.eigvalsh(t + np.diag(v))[:2]


def test_ground_level_agrees_with_an_independent_solver_and_the_perturbative_estimate():
    thetas = np.deg2rad(np.arange(0, 181, 15.0))
    r = energies_vs_angle(X_CFG, B, thetas)
    model = X_CFG.model(B)
    for s in (UPPER, LOWER):
        d = r["states"][s]
        for j in (0, 3, 6, 9):
            e = _dvr_levels(model, s, thetas[j], d["x_m"][j])
            assert e[0] == pytest.approx(d["e0_hz"][j], abs=2e-3)
            assert e[1] == pytest.approx(d["e1_hz"][j], abs=1e-2)
        # second-order perturbation theory (cubic + quartic) holds to 0.03 Hz except where the
        # well is flat-bottomed: |3,-3> at exactly 90 deg (x pair, nominal), off by 6.6 Hz there
        bad = np.abs(d["zpe_hz"] - d["zpe_estimate_hz"]) > 0.03
        assert list(np.rad2deg(thetas[bad])) == ([] if s == UPPER else [90.0])
    d = r["states"][LOWER]
    j90 = int(np.argmin(np.abs(thetas - np.pi / 2)))
    assert d["zpe_hz"][j90] - d["zpe_estimate_hz"][j90] == pytest.approx(-6.6, abs=0.1)


def test_energies_vs_angle_agree_with_the_spectrum_levels():
    site = replace(SITE, n_levels=4)
    sp = spectrum(site, max_dn=1)
    total = 0.0
    for cfg in site.axes:
        r = energies_vs_angle(cfg, B, [cfg.theta])
        lv = sp["axes"][cfg.axis]["levels"]
        for s in (UPPER, LOWER):
            assert r["states"][s]["e0_hz"][0] == pytest.approx(lv[s]["energies_hz"][0], abs=1e-3)
            assert r["states"][s]["v_min_hz"][0] == pytest.approx(lv[s]["v_min_hz"], abs=1e-3)
        total += r["carrier_hz"][0]
    assert total == pytest.approx(sp["carrier_shift_hz"], abs=3e-3)
    # the carrier contribution splits into the potential minima and the zero-point energies
    r = energies_vs_angle(X_CFG, B, [0.0, 0.5, 1.5])
    assert np.allclose(r["carrier_hz"], r["v_min_diff_hz"] + r["zpe_diff_hz"], atol=1e-6)


# ------------------------------------------------------------------ explorer extension (no window)
def _load_script(name):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_explorer_angle_model_and_drawing():
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    se, ea = _load_script("sideband_explorer"), _load_script("explorer_angle")
    s = se.default_settings()
    key = ea.energies_key(s)
    s2 = se.copy_settings(s)
    s2["pairs"]["x"]["theta_deg"] = 120.0
    assert ea.energies_key(s2) == key                         # the angle is the curves' x axis
    s2["pairs"]["x"]["er"] = 1e3
    assert ea.energies_key(s2) != key
    thetas = ea.angle_grid(60.0)
    assert thetas[0] == -180.0 and thetas[-1] == 180.0 and thetas.size == 7
    whole = ea.energies_for(se, s, thetas)
    parts = ea.merge_chunks([ea.energies_for(se, s, thetas[i:i + 3]) for i in range(0, 7, 3)], thetas)
    for st in whole["states"]:
        for k in ("x_m", "e0_hz", "zpe_hz"):
            assert np.allclose(parts["states"][st][k], whole["states"][st][k], rtol=0, atol=1e-9)
    fig = Figure(figsize=(10, 6.2), constrained_layout=True)
    FigureCanvasAgg(fig)
    ea.draw_energies(fig, se, whole, 5.4, "settings")
    assert all(ax.get_xlabel() for ax in fig.axes)            # an x label on every panel
    fig.canvas.draw()
    cache = {}
    for deg, k_expected in [(5.4, 0), (90.0, 0), (180.0, 1)]:
        s3 = se.copy_settings(s)
        s3["pairs"]["x"]["theta_deg"] = deg
        res = se.compute(s3, cache)
        assert ea.shown_lower_site(res["sp"]["axes"]["x"], LOWER) == k_expected
        for ref in ea.REFERENCES:
            ea.draw_wells(fig, se, res, ref)
            assert all(ax.get_xlabel() for ax in fig.axes)
            fig.canvas.draw()
