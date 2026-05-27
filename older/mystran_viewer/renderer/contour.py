"""
Color mapping utilities for contour rendering.
Colormaps: rainbow, jet, cool-warm, grayscale
"""

import numpy as np


# ---------------------------------------------------------------------------
# Colormaps as Nx3 float32 arrays (R,G,B) in [0,1]
# ---------------------------------------------------------------------------

def _lerp_colormap(stops, n=256):
    """Build colormap by linearly interpolating between RGB stops."""
    result = np.zeros((n, 3), dtype=np.float32)
    stops = np.array(stops, dtype=np.float32)  # shape (k, 4): t, r, g, b
    for i in range(n):
        t = i / (n - 1)
        # find segment
        for j in range(len(stops) - 1):
            t0, t1 = stops[j, 0], stops[j+1, 0]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0 + 1e-12)
                result[i] = stops[j, 1:] * (1 - alpha) + stops[j+1, 1:] * alpha
                break
    return result


COLORMAPS = {
    'rainbow': _lerp_colormap([
        [0.0,  0.0, 0.0, 1.0],   # blue
        [0.25, 0.0, 1.0, 1.0],   # cyan
        [0.5,  0.0, 1.0, 0.0],   # green
        [0.75, 1.0, 1.0, 0.0],   # yellow
        [1.0,  1.0, 0.0, 0.0],   # red
    ]),
    'jet': _lerp_colormap([
        [0.0,  0.0,  0.0,  0.5],
        [0.11, 0.0,  0.0,  1.0],
        [0.34, 0.0,  1.0,  1.0],
        [0.5,  0.0,  1.0,  0.0],
        [0.66, 1.0,  1.0,  0.0],
        [0.89, 1.0,  0.0,  0.0],
        [1.0,  0.5,  0.0,  0.0],
    ]),
    'coolwarm': _lerp_colormap([
        [0.0,  0.23, 0.30, 0.75],
        [0.5,  0.86, 0.86, 0.86],
        [1.0,  0.71, 0.02, 0.15],
    ]),
    'grayscale': _lerp_colormap([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]),
}


def map_values(values: np.ndarray, vmin: float, vmax: float,
               cmap_name: str = 'rainbow') -> np.ndarray:
    """
    Map scalar values to RGB colors.
    Returns float32 array of shape (N, 3).
    """
    cmap = COLORMAPS.get(cmap_name, COLORMAPS['rainbow'])
    n    = len(cmap)

    if vmax - vmin < 1e-30:
        t = np.zeros(len(values), dtype=np.float32)
    else:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)

    indices = (t * (n - 1)).astype(np.int32)
    indices = np.clip(indices, 0, n - 1)
    return cmap[indices]


def colormap_texture_data(cmap_name: str = 'rainbow') -> np.ndarray:
    """Return 256x1 RGB texture data (uint8) for legend bar."""
    cmap = COLORMAPS.get(cmap_name, COLORMAPS['rainbow'])
    return (cmap * 255).astype(np.uint8)


def legend_ticks(vmin: float, vmax: float, n: int = 7):
    """Return n evenly spaced tick values."""
    return np.linspace(vmin, vmax, n)
