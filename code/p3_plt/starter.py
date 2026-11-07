"""P3 starter. Implement every function marked TODO.

The transform is pointwise. That is the whole hint.
"""
import numpy as np


def transfer_curve(img_in, img_out):
    """TODO 3.1: estimate T[v] for v in 0..255, plus the pixel count supporting
    each level. Levels with no pixels must be distinguishable from levels
    that map to zero -- you need both arrays."""
    raise NotImplementedError


def fit_piecewise(T, cnt, max_seg=6):
    """TODO 3.2: fit the smallest number of segments the data justifies.

    You need a complexity penalty; a pure least-squares fit always improves
    with more segments. Read the warning in the handout about the quantisation
    floor before you tune it.

    Returns (breakpoints, slopes, intercepts, n_segments).
    """
    raise NotImplementedError


def apply_recovered(img, bps, slopes, inter):
    """TODO 3.1: build the LUT from your fit and apply it."""
    raise NotImplementedError


def support_report(cnt, **kw):
    """TODO 3.3: which intensity ranges does the input not sample well enough
    to constrain the transform? Return something human-readable."""
    raise NotImplementedError
