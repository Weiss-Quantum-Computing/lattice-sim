"""The molasses AOM power calibration: what Supertime asks for and what comes out.

Supertime converts a requested power P (the Molasses-beams channel, in the units
of the calibration file, uW) into a control voltage by inverting the calibration
file (``OM-2014-9-3.txt``). AOMrescale multiplies the file's power column so
that its maximum equals the rescale value r, so a request P at rescale r
selects the voltage at which the *unscaled* file reads P x max/r. The light
that actually comes out at that voltage follows the AOM's real curve, which
``data/measured/molasses_aom_calibration_20261029.csv`` holds as measured some
time ago ("old") and on 2026-10-29 at +100 MHz ("new").

Linear interpolation in both directions (assumed to be what the conversion VI
does). Where the file is flat (0.2-0.5 V) the inverse takes the lowest voltage.
"""

import csv
from pathlib import Path

import numpy as np

CAL_CSV = Path(__file__).resolve().parents[3] / "data" / "measured" / "molasses_aom_calibration_20261029.csv"
IMAGING_REQUEST = 5.0          # Molasses-beams value during the exposures (PGC_vapor_loading.dt4, hopping sequences)
PGC_HOLD_REQUEST = 0.6         # during the 1000 ms PGC-field hold (PGC_vapor_loading.dt4)
RESCALE_NOW = 24.0             # AOMrescale value since 2026-10-27 03:03 (aomlog.txt)
# Power-meter reading with the beam blocked (room background), subtracted from each measured curve:
# 2026-10-29 reads 0.079-0.086 uW at 0-1.1 V, "essentially with the beam blocked" (Maarten, 2026-10-29).
BACKGROUND_UW = {"new": 0.086, "old": 0.0, "supertime": 0.0}


def load_curves(path=CAL_CSV):
    """{'supertime' | 'old' | 'new': (volts, power_uw)} sorted by voltage."""
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            out.setdefault(row["curve"], []).append((float(row["volts"]), float(row["power_uw"])))
    return {k: tuple(np.array(c) for c in zip(*sorted(v))) for k, v in out.items()}


def request_to_volts(request, rescale, file_curve):
    """Voltage Supertime outputs for a request at a given rescale value (inverse of the scaled file)."""
    v, p = file_curve
    target = np.asarray(request, float) * p.max() / rescale
    keep = np.concatenate([[True], np.diff(p) > 0])      # first point of each flat run: lowest voltage
    return np.interp(target, p[keep], v[keep])


def power_at(volts, curve, background_uw=0.0):
    """Measured power (uW) at a control voltage, linear interpolation, background subtracted (>= 0)."""
    v, p = curve
    return np.clip(np.interp(volts, v, p) - background_uw, 0.0, None)


def delivered(request, rescale, curves, measured="new"):
    """(volts, delivered power uW, nominal uW) for a request: nominal = what the scaled file promises;
    the delivered power has the meter background (BACKGROUND_UW) subtracted."""
    volts = request_to_volts(request, rescale, curves["supertime"])
    return volts, power_at(volts, curves[measured], BACKGROUND_UW.get(measured, 0.0)), np.asarray(request, float)


def relative_power(rescale, reference=RESCALE_NOW, request=IMAGING_REQUEST, curves=None, measured="new"):
    """Delivered power at `rescale` relative to that at `reference`, same request (the 1/r rule is only
    what the file promises)."""
    curves = curves or load_curves()
    _, p, _ = delivered(request, np.asarray(rescale, float), curves, measured)
    _, p0, _ = delivered(request, reference, curves, measured)
    return p / p0


def supertime_file_lines(curve, offset_uw=0.0):
    """Rows 'volts<TAB>power' for a Supertime calibration file from a measured curve; the power is made
    non-decreasing and starts at 0 (Supertime's files do)."""
    v, p = curve
    p = np.maximum.accumulate(np.clip(p - offset_uw, 0.0, None))
    p[0] = 0.0
    # of a flat run keep its first and last point, so the inverse interpolates from the threshold
    keep = np.ones(len(p), bool)
    keep[1:-1] = ~((p[1:-1] == p[:-2]) & (p[1:-1] == p[2:]))
    return [f"{a:g}\t{b:.6g}" for a, b in zip(v[keep], p[keep])]
