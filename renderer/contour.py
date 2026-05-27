"""
Color mapping utilities for contour rendering.
Uses lazy numpy initialization to avoid module-level numpy crashes on
experimental Python 3.13 Windows MINGW builds.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Build colormaps lazily (avoid module-level np.zeros/np.array on broken numpy)
# ---------------------------------------------------------------------------

_STOPS = {
    'rainbow': [
        [0.0,  0.0, 0.0, 1.0],
        [0.25, 0.0, 1.0, 1.0],
        [0.5,  0.0, 1.0, 0.0],
        [0.75, 1.0, 1.0, 0.0],
        [1.0,  1.0, 0.0, 0.0],
    ],
    'jet': [
        [0.0,  0.0,  0.0,  0.5],
        [0.11, 0.0,  0.0,  1.0],
        [0.34, 0.0,  1.0,  1.0],
        [0.5,  0.0,  1.0,  0.0],
        [0.66, 1.0,  1.0,  0.0],
        [0.89, 1.0,  0.0,  0.0],
        [1.0,  0.5,  0.0,  0.0],
    ],
    'coolwarm': [
        [0.0,  0.23, 0.30, 0.75],
        [0.5,  0.86, 0.86, 0.86],
        [1.0,  0.71, 0.02, 0.15],
    ],
    'grayscale': [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ],
}

_CACHE: dict = {}   # lazy cache


def _build(name: str, n: int = 256):
    stops = _STOPS[name]
    result = [[0.0, 0.0, 0.0]] * n
    for i in range(n):
        t = i / (n - 1)
        for j in range(len(stops) - 1):
            t0, t1 = stops[j][0], stops[j+1][0]
            if t0 <= t <= t1 + 1e-9:
                a = (t - t0) / (t1 - t0 + 1e-12)
                r = stops[j][1]*(1-a) + stops[j+1][1]*a
                g = stops[j][2]*(1-a) + stops[j+1][2]*a
                b = stops[j][3]*(1-a) + stops[j+1][3]*a
                result[i] = [r, g, b]
                break
    return np.array(result, dtype=np.float32)


def _get(name: str):
    if name not in _CACHE:
        stops = _STOPS.get(name, _STOPS['rainbow'])
        _CACHE[name] = _build(name)
    return _CACHE[name]


# Public dict-like access — lazy
class _ColormapDict:
    def __getitem__(self, key):
        return _get(key if key in _STOPS else 'rainbow')
    def get(self, key, default=None):
        return _get(key if key in _STOPS else 'rainbow')
    def keys(self):
        return _STOPS.keys()
    def __contains__(self, key):
        return key in _STOPS


COLORMAPS = _ColormapDict()


def map_values(values: np.ndarray, vmin: float, vmax: float,
               cmap_name: str = 'rainbow') -> np.ndarray:
    cmap = _get(cmap_name)
    n    = len(cmap)
    if vmax - vmin < 1e-30:
        t = np.zeros(len(values), dtype=np.float32)
    else:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    indices = np.clip((t * (n - 1)).astype(np.int32), 0, n - 1)
    return cmap[indices]


def legend_ticks(vmin: float, vmax: float, n: int = 7):
    return np.linspace(vmin, vmax, n)
