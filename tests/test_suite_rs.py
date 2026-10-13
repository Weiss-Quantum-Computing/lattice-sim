"""Shape-robust suite ramp RS25: shape, export format and engine agreement."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from design_suite_shape_robust import slew_par, theta_of  # noqa: E402

DESIGN = ROOT / "data" / "suite_rs_K0.94_2500_K12_s0.json"


def test_theta_of_endpoints_and_monotonic():
    c = np.array(json.loads(DESIGN.read_text(encoding="utf-8"))["c"])
    th = theta_of(c, n=1251)
    assert th[0] == pytest.approx(0, abs=1e-9) and th[-1] == pytest.approx(180, abs=1e-9)
    assert np.all(np.diff(th) >= -1e-12)
    # smooth7 time warp: zero velocity at both ends
    assert abs(th[1] - th[0]) < 1e-6 and abs(th[-1] - th[-2]) < 1e-6
    assert slew_par(th, 2500.0) < 6.0          # V/us per EOM, PAR (design: 5.65)


def test_export_round_trip(tmp_path, monkeypatch):
    import export_suite_target as E
    from score_suite import read_pair, rise_end_index
    monkeypatch.chdir(ROOT)
    E.main(str(DESIGN), "RS25", str(tmp_path), "test")
    t, th, k0, _ = read_pair(tmp_path, "RS25")
    assert k0 == 0.94
    assert t[1] - t[0] == pytest.approx(2.0) and t.size == 3201
    rise = theta_of(np.array(json.loads(DESIGN.read_text(encoding="utf-8"))["c"]), n=1251)
    i0 = 100                                     # 200 us lead
    assert np.allclose(np.rad2deg(th[i0:i0 + rise.size]), rise, atol=1e-5)
    assert abs(rise_end_index(th) - (i0 + rise.size - 1)) <= 2     # flat to 1e-9 just before the top
    assert np.allclose(th, th[::-1], atol=1e-12)   # mirrored fall, symmetric leads
    pred = json.loads((tmp_path / "RS25_prediction.json").read_text(encoding="utf-8"))
    assert max(pred["lattice_sim_rise_1mP0_worst_x085_115"].values()) < 1e-3


def test_split_step_agrees_with_gaussian():
    """Recorded validation (scripts/validate_suite_rs.py): full anharmonic tables vs harmonic engine."""
    path = ROOT / "data" / "validate_suite_rs.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    for stem in ("RS25", "S25"):
        for key, v in res[stem].items():
            if v["gauss"] > 1e-4:
                assert 0.6 < v["split"] / v["gauss"] < 1.4, (stem, key, v)
    assert max(v["split"] for v in res["RS25"].values()) < 1e-3
    assert max(v["split"] for v in res["S25"].values()) > 1e-2
