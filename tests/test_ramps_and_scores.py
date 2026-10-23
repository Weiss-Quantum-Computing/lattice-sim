"""The ramp optimizer's shape constraints and the suite scorer (``scripts/score_suite.py``)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lattice_sim import cache
from lattice_sim.control.ramp_optimizer import V_PI, RampProblem, save_ramp
from lattice_sim.motion.fast_gaussian import propagate_scenarios

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class Static:
    """A trap that does not move or change: nothing is excited."""
    name = "static"

    def x0(self, theta):
        return np.zeros_like(theta)

    def omega(self, theta):
        return np.full_like(theta, 2 * np.pi * 10e3)


def test_projection_meets_the_endpoint_conditions():
    prob = RampProblem([Static()], duration_us=500.0, terms=8, dt_us=0.5)
    rng = np.random.default_rng(1)
    a = prob.project(rng.normal(0, 0.05, prob.terms))
    A, b = prob.endpoint_matrix()
    assert A @ a == pytest.approx(b, abs=1e-12)
    s = prob.s(a)
    assert s[0] == pytest.approx(0.0, abs=1e-12) and s[-1] == pytest.approx(1.0, abs=1e-12)
    sdot = prob.sdot(a)
    assert sdot[0] == pytest.approx(0.0, abs=1e-12) and sdot[-1] == pytest.approx(0.0, abs=1e-12)
    # third derivative (jerk) at both ends from the sine series, independently of A
    k = np.arange(1, prob.terms + 1)
    w = np.pi * k / prob.duration_us
    jerk_0, jerk_t = -np.sum(a * w ** 3), -np.sum(a * w ** 3 * np.cos(np.pi * k))
    assert jerk_0 == pytest.approx(0.0, abs=1e-15) and jerk_t == pytest.approx(0.0, abs=1e-15)
    assert prob.project(a) == pytest.approx(a, abs=1e-14)           # a projection


def test_saved_ramp_is_monotonic_and_ends_at_v_pi(tmp_path):
    prob = RampProblem([Static()], duration_us=500.0, terms=4)
    a = prob.project(np.zeros(prob.terms))
    path = tmp_path / "ramp.csv"
    save_ramp(path, prob, a)
    data = np.genfromtxt(path, delimiter=",", names=True)
    assert np.all(np.diff(data["voltage_V"]) >= 0)
    assert data["voltage_V"][0] == pytest.approx(0.0, abs=1e-9) and data["voltage_V"][-1] == pytest.approx(V_PI)
    assert prob.band_min(a) == pytest.approx([1.0], abs=1e-9)           # a static trap is never excited


def _score_suite():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("score_suite", SCRIPTS / "score_suite.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPTS))
    return mod


def _write_pair(folder, stem, t_us, v1, v2, v90=(5128.3, 5137.4), depth="1/4"):
    for ch, v, v90_ch in (("X1", v1, v90[0]), ("X2", v2, v90[1])):
        with open(folder / f"target_{stem}{ch}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# synthetic suite target {stem}\n")
            fh.write(f"# scaled to this channel's V90 {v90_ch} V\n")
            fh.write(f"# designed for {depth} lattice intensity\n")
            fh.write("time_us,voltage_V\n")
            for t, x in zip(t_us, v):
                fh.write(f"{t:.1f},{x:.6f}\n")


def test_suite_pair_reading_and_harmonic_score(tmp_path):
    ss = _score_suite()
    t = np.arange(0.0, 3000.0 + 1e-9, 2.0)
    rise = np.clip(t / 1000.0, 0, 1)                                   # 1 ms rise, then hold
    s = np.sin(0.5 * np.pi * rise) ** 2
    _write_pair(tmp_path, "T1", t, 5128.3 * s, 5137.4 * s)
    t_us, theta, k0, title = ss.read_pair(tmp_path, "T1")
    assert k0 == 0.94 and title.startswith("synthetic")
    assert theta[-1] == pytest.approx(np.pi, rel=1e-9)                 # V90 + V90 = 180 deg
    assert theta == pytest.approx(np.pi * s, rel=1e-6, abs=1e-9)
    assert ss.rise_end_index(theta) == int(np.argmax(t >= 1000.0))
    # a static trap: no excitation at all
    from scipy.interpolate import CubicSpline
    th = np.linspace(0, np.pi, 11)
    static = {"F3": (CubicSpline(th, np.zeros_like(th)), CubicSpline(th, np.full_like(th, 10e3)))}
    out = ss.score_harmonic(t_us, theta, static, k0)
    assert out["worst_robust"]["rise"] == pytest.approx(0.0, abs=1e-9)
    # a moving trap: the engine is the exact Gaussian propagation, checked directly
    moving = {"F3": (CubicSpline(th, 2.4e-6 * np.sin(th / 2) ** 2), CubicSpline(th, np.full_like(th, 10e3)))}
    out = ss.score_harmonic(t_us, theta, moving, k0)
    tg = np.arange(t_us[0], t_us[-1] + 1e-9, ss.DT_US)
    thg = np.interp(tg, t_us, theta)
    p0 = propagate_scenarios(tg, moving["F3"][0](thg)[None, :], 2 * np.pi * 10e3 * k0 * np.ones((1, tg.size)))
    assert out["per_scenario"]["F3_x1.00"]["end"] == pytest.approx(1 - p0[0, -1], rel=1e-9, abs=1e-15)


def test_suite_header_errors_and_empty_folders(tmp_path):
    ss = _score_suite()
    t = np.arange(0.0, 20.0, 2.0)
    _write_pair(tmp_path, "BAD", t, np.zeros_like(t), np.zeros_like(t), depth="1/3")
    with pytest.raises(ValueError, match="target_BADX1.csv"):
        ss.read_pair(tmp_path, "BAD")
    with pytest.raises(ValueError, match="does not exist"):
        ss.find_stems(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no target pairs"):
        ss.find_stems(tmp_path / "empty")
    with pytest.raises(ValueError, match="for the stems"):
        ss.find_stems(tmp_path, ["S25"])
    assert ss.find_stems(tmp_path) == ["BAD"]


def test_suite_command_line(tmp_path, monkeypatch):
    ss = _score_suite()
    monkeypatch.chdir(tmp_path)
    folder, stems, split = ss.parse_args(["suite", "--stems", "S25,N15"])
    assert folder == (tmp_path / "suite").resolve() and stems == ["S25", "N15"] and not split
    assert ss.parse_args(["suite", "--stems=S25", "--split"])[1:] == (["S25"], True)


def test_ellipticity_phase_scales_with_the_depth():
    """Both light-shift terms scale as the depth, k0^2 (the ellipticity term went as k0)."""
    ss = _score_suite()
    t = np.arange(0.0, 1000.0 + 1e-9, 2.0)
    theta = np.full_like(t, 0.5)
    th_e = np.linspace(0, np.pi, 7)
    e_lin, e_ell = np.full(7, 100.0), np.full(7, 30.0)                 # Hz
    for k0 in (0.67, 0.94, 1.33):
        ph = ss.light_shift_phases(t, theta, k0, th_e, e_lin, e_lin + e_ell)
        dt = np.gradient(t) * 1e-6
        assert ph["phase_rad_linear"] == pytest.approx(2 * np.pi * 100.0 * k0 ** 2 * dt.sum(), rel=1e-12)
        assert ph["phase_rad_measured_er"] - ph["phase_rad_linear"] == pytest.approx(
            2 * np.pi * 30.0 * k0 ** 2 * dt.sum(), rel=1e-12)


def test_cache_records_its_inputs(tmp_path):
    path = tmp_path / "c.npz"
    inputs = dict(er_csv_digest="abc", b=np.array([1.0, 0.5, 0.0]), n=3, er=np.inf)
    assert cache.load(path, inputs) is None
    cache.save(path, inputs, y=np.arange(3.0))
    assert cache.load(path, dict(inputs))["y"].tolist() == [0.0, 1.0, 2.0]
    assert cache.load(path, dict(inputs, n=4)) is None                   # another input: rebuilt
    assert cache.load(path, dict(inputs, b=np.array([1.0, 0.5, 1e-9]))) is None
    np.savez(tmp_path / "old.npz", y=np.arange(3.0))                     # no record: rebuilt
    assert cache.load(tmp_path / "old.npz", inputs) is None
    assert cache.stored_inputs(path)["n"] == 3


def test_split_tables_are_checked_against_the_extinction_csv(tmp_path, monkeypatch):
    from lattice_sim.motion.transport1d import PotentialTable
    ss = _score_suite()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    csv_a, csv_b = tmp_path / "a.csv", tmp_path / "b.csv"
    csv_a.write_text("method,er\ndip,3000\n", encoding="utf-8")
    csv_b.write_text("method,er\ndip,2000\n", encoding="utf-8")
    monkeypatch.setattr(ss, "ER_CSV", str(csv_a))
    assert ss.split_tables() is None                                    # no tables at all
    table = PotentialTable(np.array([0.0, np.pi]), np.linspace(-1e-6, 1e-6, 5), np.zeros((2, 5)))
    for f in ("3m-3", "4m-3"):
        table.save(f"data/x_ERmeas133_h-1_10kHz_F{f}.npz",
                   inputs=dict(er_csv="b.csv", er_csv_digest=cache.file_digest(csv_b)))
    assert ss.split_tables() is None                                    # built from another CSV
    for f in ("3m-3", "4m-3"):
        table.save(f"data/x_ERmeas133_h-1_10kHz_F{f}.npz",
                   inputs=dict(er_csv="a.csv", er_csv_digest=cache.file_digest(csv_a)))
    tables = ss.split_tables()
    assert set(tables) == {"F3", "F4"} and tables["F3"].inputs["er_csv"] == "a.csv"
    table.save("data/x_ERmeas133_h-1_10kHz_F3m-3.npz")                  # an old table without a record
    assert PotentialTable.load("data/x_ERmeas133_h-1_10kHz_F3m-3.npz").inputs is None
    assert ss.split_tables() is not None                                # used, with a printed note
