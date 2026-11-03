"""scripts/camera_runs.py on a synthetic CameraGui run with known hops and losses."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("camera_runs", Path(__file__).resolve().parents[1] / "scripts" / "camera_runs.py")
cr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cr)


def _write_run(folder, shots=30, seed=1):
    rng = np.random.default_rng(seed)
    occ1 = (rng.random((shots, 9, 9, 9)) < 0.3).astype(int)
    occ2 = occ1.copy()
    hops = losses = 0
    for i in range(shots):                      # one loss and one hop in plane 4 of every shot
        z = 4
        full = np.argwhere(occ1[i, z] == 1)[0]
        empty = np.argwhere(occ1[i, z] == 0)[0]
        occ2[i, z, full[0], full[1]] = 0
        occ2[i, z, empty[0], empty[1]] = 1
        hops += 1
        losses += 1
    occ = np.stack([occ1, occ2], 1)
    h = np.where(occ == 1, rng.normal(17000, 1500, occ.shape), rng.normal(0, 800, occ.shape))
    np.savetxt(folder / "001.txt.hist", h.reshape(-1))
    with open(folder / "001.txt.atom", "w", encoding="utf-8") as fh:
        fh.write("N = %d\nthresholds = 9000 11000 12000 14000 9000 11000 12000 14000\n" % shots)
        for i in range(shots):
            fh.write("Run = %d\n" % (i + 1))
            fh.write("Before = " + " ".join(map(str, occ1[i].ravel())) + "\n")
            fh.write("After = " + " ".join(map(str, occ2[i].ravel())) + "\n")
    return hops, losses, int(occ1[:, 1:8].sum())


def test_counts_brightness_and_moves(tmp_path):
    hops, losses, atoms = _write_run(tmp_path)
    r = cr.analyse(tmp_path, "001")
    assert r["atoms1"] == atoms
    assert r["hops"] == hops and r["losses"] == losses
    assert abs(r["bright"] - 17000) < 500
    assert r["sep"] > 8 and r["miscall"] < 1e-6


def test_scan_points_and_bad_shot(tmp_path):
    _write_run(tmp_path, shots=20)
    r = cr.analyse(tmp_path, "001", scan=(0.0, 1.0, 5), bad_hops=0)   # every shot has one hop: all dropped
    assert r["shots_used"] == 0
    r = cr.analyse(tmp_path, "001", scan=(0.0, 1.0, 5))
    assert [p["shots"] for p in r["points"]] == [4] * 5
