"""Caches in .npz files that remember the inputs they were built from.

A cache file stores, next to its arrays, a JSON record of its inputs (file
digests of the measured data it read, fields, trap frequencies, grids). On
load the record is compared with the caller's inputs; a file without a
record, or with different inputs, counts as missing, so editing an input
(a new extinction CSV, another bias field) rebuilds the cache instead of
silently reusing the old one.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

INPUTS_KEY = "cache_inputs"


def file_digest(path):
    """SHA-256 (first 16 hex digits) of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return repr(float(value))                  # repr: exact, and 'inf' / 'nan' survive JSON
    return value


def inputs_record(inputs):
    """The canonical JSON text of an inputs dict (what is stored and compared)."""
    return json.dumps(_plain(inputs), sort_keys=True)


def load(path, inputs):
    """The cache's arrays as a dict, or None when the file is missing, has no
    inputs record, or was built from other inputs."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        if INPUTS_KEY not in d.files or str(d[INPUTS_KEY]) != inputs_record(inputs):
            return None
        return {k: d[k] for k in d.files if k != INPUTS_KEY}


def save(path, inputs, **arrays):
    """Write the arrays and the inputs record (the folder must exist)."""
    np.savez(path, **{INPUTS_KEY: np.array(inputs_record(inputs))}, **arrays)


def stored_inputs(path):
    """The inputs dict a cache file was built from, or None."""
    with np.load(path, allow_pickle=False) as d:
        if INPUTS_KEY not in d.files:
            return None
        return json.loads(str(d[INPUTS_KEY]))
