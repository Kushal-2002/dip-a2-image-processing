"""P4 starter. Implement every function marked TODO.

Dataset map (relative to the bundle root):
  4.1  images/p4/equalization/input.png
  4.2  images/p4/matching/{source,reference}.png
       images/p4/specification/source.png
       images/p4/colour_source.png
  4.3  images/p4/local_regions.png
  4.4  images/p4/exposure/ev0.png ... ev4.png
"""
import numpy as np


def hist(img, bins=256):
    return np.bincount(img.ravel(), minlength=bins).astype(np.float64)


def cdf(h):
    """TODO 4.1: normalised cumulative histogram."""
    raise NotImplementedError


def equalise(img):
    """TODO 4.1: CDF-based global equalisation."""
    raise NotImplementedError


def specify(img, target_hist):
    """TODO 4.2: match img's histogram to target_hist."""
    raise NotImplementedError


def wasserstein1(h1, h2):
    """TODO 4.2: W1 distance between two histograms. Your scoring metric."""
    raise NotImplementedError


def ahe(img, tiles=8):
    """TODO 4.3: plain tiled AHE. No clipping, no interpolation. This one is
    SUPPOSED to look bad -- that is the point."""
    raise NotImplementedError


def clahe(img, tiles=8, clip=3.0, bins=256):
    """TODO 4.3: clip at `clip` x mean bin height, redistribute the excess,
    and bilinearly interpolate between the four surrounding tile LUTs.

    Watch the tile-centre offset. That is where the marks go."""
    raise NotImplementedError


def apply_on_luma(rgb, fn):
    """TODO 4.2: run a greyscale operator on luma only, preserving hue."""
    raise NotImplementedError


def auto_correct(rgb):
    """TODO 4.4: no parameters. Estimate what you need from the image."""
    raise NotImplementedError
