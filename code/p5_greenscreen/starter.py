"""P5 starter. Implement every function marked TODO."""
import numpy as np


def to_ycbcr(rgb):
    """TODO 5.1: BT.601 conversion."""
    raise NotImplementedError


def key_naive_rgb(rgb, thresh=60):
    """TODO 5.1: green dominance in raw RGB. Expected to fail on the unevenly
    lit backing -- show where."""
    raise NotImplementedError


def estimate_key_colour(rgb, border=24):
    """TODO 5.2: estimate the backing chroma from the frame. Do not hard-code
    a green value; the graded footage varies."""
    raise NotImplementedError


def key_soft(rgb, key_cbcr=None, t_in=None, t_out=None):
    """TODO 5.2: fractional alpha from chroma distance. Choose t_in and t_out
    from your own data -- look at the distance distribution for pure backing
    versus pure foreground before you pick numbers."""
    raise NotImplementedError


def suppress_spill(rgb, alpha, strength=1.0):
    """TODO 5.2: remove green bounce. Should scale with (1 - alpha). Why?"""
    raise NotImplementedError


def composite(fg, bg, alpha, spill=True):
    """TODO 5.2: I = a*F + (1-a)*B."""
    raise NotImplementedError


class ChromaLUT:
    """TODO 5.3: quantise (Cb,Cr) so keying is one table lookup per pixel."""

    def __init__(self, key_cbcr, t_in=None, t_out=None, bins=128):
        raise NotImplementedError

    def alpha(self, rgb):
        raise NotImplementedError


# ---- provided metrics: do not modify
def sad(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / 1000.0)


def mse_alpha(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))


def grad_error(a, b):
    from scipy.ndimage import gaussian_filter
    ga = np.hypot(*np.gradient(gaussian_filter(a.astype(np.float64), 1.0)))
    gb = np.hypot(*np.gradient(gaussian_filter(b.astype(np.float64), 1.0)))
    return float(np.abs(ga - gb).sum() / 1000.0)


def temporal_flicker(alphas):
    if len(alphas) < 2:
        return 0.0
    total = 0.0
    count = 0
    prev = np.asarray(alphas[0], dtype=np.float32)
    for alpha in alphas[1:]:
        cur = np.asarray(alpha, dtype=np.float32)
        total += float(np.abs(cur - prev).sum(dtype=np.float64))
        count += cur.size
        prev = cur
    return total / count
