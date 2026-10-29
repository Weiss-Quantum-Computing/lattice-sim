"""Tests for lattice_sim.molasses (semiclassical Monte Carlo PGC model).

Fast checks (level structure, geometry) run always; the Monte Carlo checks
use small ensembles and loose tolerances (statistical); the full-statistics
versions are in scripts/molasses_validate.py and docs/findings.md (2026-10-29).
"""

import numpy as np
import pytest

pytest.importorskip("sympy")
numba = pytest.importorskip("numba")

from lattice_sim.molasses import beams, levels, model  # noqa: E402


def test_decomposition_exact():
    tr = levels.cs_d2_f4()
    assert tr.check() < 1e-12
    assert levels.j_half_to_three_half().check() < 1e-12


def test_cs_branching_and_strengths():
    tr = levels.cs_d2_f4()
    br = {lev.label: lev.branch_to_g for lev in tr.levels}
    assert br["F'=5"] == pytest.approx(1.0, abs=1e-12)
    assert br["F'=4"] == pytest.approx(7 / 12, abs=1e-12)      # Steck: F'=4 -> F=4
    assert br["F'=3"] == pytest.approx(1 / 4, abs=1e-12)       # Steck: F'=3 -> F=4
    s = np.array([np.einsum("qem,qem->m", lev.d, lev.d)[0] for lev in tr.levels])
    np.testing.assert_allclose(s / s.sum(), [11 / 18, 7 / 24, 7 / 72], atol=1e-12)


def test_sigma_plus_light_shift_cgs():
    tr = levels.cs_d2_f4((5,))
    m = tr.m_exact(tr.levels[0], levels.E_SPH[2])           # pure sigma+ along z
    diag = np.diag(m).real
    assert diag[-1] == pytest.approx(1.0)                    # |4,4> -> |5',5'>
    assert diag[0] == pytest.approx(1 / 45)                  # |4,-4> -> |5',-3'>


def test_geometry_lin_perp_lin_and_transverse():
    for g in (beams.lab_brewster(), beams.orthogonal_linperp(), beams.one_d_linperp()):
        k, e = g["k_hat"], g["eps"]
        for i in range(0, len(k), 2):
            np.testing.assert_allclose(k[i], -k[i + 1], atol=1e-12)
            assert abs(np.vdot(e[i], e[i + 1])) < 1e-12        # orthogonal linear polarizations
    k = beams.lab_brewster()["k_hat"]
    assert np.degrees(np.arccos(k[0, 2])) == pytest.approx(beams.BREWSTER_DEG)


def test_dragged_atom_friction_and_diffusion_dct():
    """1D lin-perp-lin J = 1/2 -> 3/2 at delta = -10 Gamma against Dalibard & Cohen-Tannoudji (1989):
    alpha = 3 hbar k^2 |delta|/Gamma, D = (3/8) V0^2 k^2/Gamma' (full statistics: 0.94(5), 0.97)."""
    I, det = 2.0, -52.34
    ref = model.dct_linperp(I, det)
    k = 2 * np.pi / levels.LAMBDA_D2
    base = dict(transition="jhalf", levels=(5,), geometry="1d", lattice=False, doppler=False,
                motion=(False, False, False), drag=True, t_total_ms=1.6, t_burn_ms=0.3, n_atoms=256, seed=3)
    v = 0.05 * ref["gamma_p"] / k
    f = []
    for sgn in (1, -1):
        r = model.run(model.Config(intensity=I, detuning=det, drag_v=(0, 0, sgn * v), **base))
        f.append(np.mean(r.p_transfer[:, 2] / r.t_rec_s))
        assert r.scatter_rate == pytest.approx(ref["scatter"], rel=0.04)
    alpha = -(f[0] - f[1]) / (2 * v)
    assert alpha == pytest.approx(ref["alpha"], rel=0.25)
    r = model.run(model.Config(intensity=I, detuning=det, drag_v=(0, 0, 0.0), **base))
    p2 = r.p_transfer_2[:, 2]
    p1 = r.p_transfer[:, 2] - p2
    d = np.var(p1 - p2) / (2 * r.t_rec_s.mean())
    assert d == pytest.approx(ref["diffusion"], rel=0.25)


def test_step_size_convergence_in_lattice():
    common = dict(intensity=0.5, detuning=-40.0, n_atoms=96, t_total_ms=1.2, t_burn_ms=0.4, seed=21)
    a = model.run(model.Config(dt_phase=0.25, **common))
    b = model.run(model.Config(dt_phase=0.5, **common))
    err = np.hypot(a.t_uk_err, b.t_uk_err).mean()
    assert abs(a.t_mean - b.t_mean) < 4 * err
    # the rate differs from atom to atom (local molasses polarization), so compare within the statistics
    assert abs(a.scatter_rate - b.scatter_rate) < 4 * np.hypot(a.scatter_rate_err, b.scatter_rate_err)
    assert np.all(a.hops_per_s == 0)


def _row(det, inten, t, b=(0.0, 0.0, 0.0), tau=0.1):
    return dict(detuning=det, intensity=inten, b_mg=list(b), depth_scale=1.0, t_uk=[t, t, t], t_uk_err=[0.1] * 3,
                tau_e_ms=[tau] * 3, scatter_rate=1e5 * inten, depump_frac=0.01, light_shift_uk=-10.0,
                barrier_uk=[268.0, 192.0, 115.0])


def test_tables_interpolation_nodes_and_power_law():
    from lattice_sim.molasses import tables
    rows = [_row(d, i, 6.0 * i / abs(d) * 51) for d in (-51.0, -20.0) for i in (0.1, 0.4, 1.6)]
    tab = tables.Tables(rows)
    assert tab.complete
    np.testing.assert_allclose(tab.temperature(-51.0, 0.4), 2.4, rtol=1e-12)        # a node
    np.testing.assert_allclose(tab.temperature(-51.0, 0.2), 1.2, rtol=1e-12)        # T ~ I is exact in log-log
    np.testing.assert_allclose(tab.temperature(-51.0, 5.0), tab.temperature(-51.0, 1.6))   # clamped
    assert not tab.inside(-51.0, 5.0)
    h = tab.hop_rate(-51.0, 1.6)
    ref = model.arrhenius_hop_rate(9.6, 0.1, np.array([268.0, 192.0, 115.0]))
    np.testing.assert_allclose(h, ref, rtol=1e-12)


def test_crossing_field_and_survival():
    from lattice_sim.molasses import tables
    b = np.array([0.0, 50.0, 100.0, 200.0])
    t = 4.0 * (1 + (b / 100.0) ** 2)
    assert tables.crossing_field_mg(b, t) == pytest.approx(100.0)
    assert tables.crossing_field_mg(b, t, factor=10.0) == np.inf
    assert tables.survival([100.0, 0.0, 0.0], 10.0)[0] == pytest.approx(np.exp(-1.0))
    rows = [_row(-51.0, 0.35, tt, b=(bb, 0, 0)) for bb, tt in zip(b, t)]
    tab = tables.Tables([_row(-51.0, 0.35, 4.0)], rows)
    assert tab.doubling_field(-51.0, 0.35, 0) == pytest.approx(100.0)
    assert tab.field_curve(-51.0, 0.35, 1)[0].tolist() == [0.0]


def _load_script(name):
    import importlib.util
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(scripts))
    return mod


def test_molasses_window_extension_and_drawing_headless():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from lattice_sim.molasses import tables
    ext = _load_script("explorer_molasses")
    se = _load_script("sideband_explorer")
    d = ext.defaults_from_settings(se.default_settings(), se)
    site = se.site_from(se.default_settings())
    assert d["trap_khz"][0] == pytest.approx(site.axes[0].trap_frequency_hz * 1e-3)
    assert "explorer_molasses" in se.EXTENSIONS
    mx = _load_script("molasses_explorer")
    from types import SimpleNamespace

    def finish(ax, xlabel=None, ylabel=None, title=None, grid=True):
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
    lp = SimpleNamespace(finish=finish, legend=lambda ax, **kw: ax.legend(**{k: v for k, v in kw.items()
                                                                              if k != "loc"}),
                         note=lambda ax, text, where="top-left": ax.text(0, 0, text),
                         placeholder=lambda fig, text: (fig.clear(), fig.text(0.5, 0.5, text)))   # labplot needs Tk
    rows = [_row(dd, i, 6.0 * i / abs(dd) * 51) for dd in (-51.0, -20.0) for i in (0.1, 0.4, 1.6)]
    frows = [_row(-51.0, 0.4, 2.4 * (1 + (b / 100) ** 2), b=(b, 0, 0)) for b in (0.0, 100.0, 200.0)]
    tab = tables.Tables(rows, frows)
    s = dict(mx.DEFAULTS, detuning=-30.0, intensity=0.5)
    for draw in (lambda f: mx.draw_intensity(f, tab, s, None, lp), lambda f: mx.draw_field(f, tab, s, None, lp),
                 lambda f: mx.draw_live(f, None, lp), lambda f: mx.draw_intensity(f, None, s, None, lp)):
        fig = Figure()
        draw(fig)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig).draw()
    cfg = mx.live_config(s)
    assert cfg.t_total_ms - cfg.t_burn_ms == pytest.approx(s["live_ms"])


def test_total_saturation_limit():
    g = levels.GAMMA_D2 / (2 * np.pi * 1e6)
    assert model.total_saturation(levels.ISAT_CYCLING_MW_CM2, 0.0, n_beams=1) == pytest.approx(1.0)
    assert model.total_saturation(1.0, -g / 2) == pytest.approx(6 / levels.ISAT_CYCLING_MW_CM2 / 2)
    assert model.total_saturation(0.7, -51.0) < model.MAX_SATURATION < model.total_saturation(1.0, -10.0)
