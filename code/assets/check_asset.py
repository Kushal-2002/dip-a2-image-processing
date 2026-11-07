"""Check a candidate image against what its assignment slot actually needs.

An image can look excellent and still break the exercise it is used for. The
P3 transform is unrecoverable if the source does not sample enough intensity
levels; the P5 matting problem collapses to a hard threshold if the footage
has too little fractional alpha. Both failures are invisible by eye.

Usage
-----
  python assets/check_asset.py --role plt_source  candidate.png
  python assets/check_asset.py --role greenscreen frames/*.png
  python assets/check_asset.py --all data/          # audit everything in place

Roles: conv_base, cover_textured, cover_smooth, plt_source, hist_lowcontrast,
       hist_natural, hist_clahe, greenscreen, background
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

VERDICTS = []

# When non-None, report() collects into this list instead of printing, so
# fetch_images.py can audition a download candidate quietly.
_SINK = None


def _grey(a):
    return np.asarray(Image.fromarray(a).convert("L")).astype(np.float64) if a.ndim == 3 \
        else a.astype(np.float64)


def margin(value, thr, want):
    """Fractional distance from the threshold. Negative means failing.

    want="min": value must be >= thr.  want="max": value must be <= thr.
    Used to rank candidate images by how far clear of the bar they sit, not
    merely whether they clear it.
    """
    if thr == 0:
        return 0.0
    return (value - thr) / abs(thr) if want == "min" else (thr - value) / abs(thr)


def report(ok, label, detail, fix="", value=None, thr=None, want=None):
    if _SINK is not None:
        m = None if value is None else margin(value, thr, want)
        _SINK.append((ok, label, detail, fix, m))
        return
    VERDICTS.append(ok)
    print(f"  [{'OK ' if ok else 'BAD'}] {label}: {detail}")
    if not ok and fix:
        print(f"        -> {fix}")


def evaluate(role, arr, path="<memory>"):
    """Run one role's checks on an array.

    Returns (all_ok, [(ok, label, detail, fix), ...]). Prints nothing and does
    not touch the CLI's pass/fail tally.
    """
    global _SINK
    prev, _SINK = _SINK, []
    try:
        ROLES[role](arr, path)
        results = _SINK
    finally:
        _SINK = prev
    margins = [r[4] for r in results if len(r) > 4 and r[4] is not None]
    return (all(ok for ok, *_ in results), results,
            min(margins) if margins else 0.0)


# ---------------------------------------------------------------- role checks
def check_conv_base(a, path):
    h, w = a.shape[:2]
    report(min(h, w) >= 2048, "size", f"{w}x{h}",
           "need >=2048 on the short side for the N-sweep to reach 2048^2",
           value=min(h, w), thr=2048, want="min")
    g = _grey(a)
    hf = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
    report(hf > 6.0, "high-frequency energy", f"mean |grad| = {hf:.1f}",
           "too smooth: low-rank approximation will look better than it is",
           value=hf, thr=6.0, want="min")


def check_cover_textured(a, path):
    g = _grey(a)
    report(min(g.shape) >= 1024, "size", f"{g.shape[1]}x{g.shape[0]}",
           value=min(g.shape), thr=1024, want="min")
    hf = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
    report(hf > 12.0, "texture energy", f"mean |grad| = {hf:.1f}",
           "not textured enough to hide an LSB payload convincingly",
           value=hf, thr=12.0, want="min")
    # Embedding headroom. embed_robust walks 8x8 blocks in row-major order, and
    # the covers are 1024 wide, so all 128 bits of the robustness test land in
    # the FIRST block row -- the top 8 pixel rows of the image. If that strip is
    # blown-out sky, a +/-3 block-mean shift has nowhere to go: degrade_gaussian
    # clips at 255, which drags the block mean down and flips the bit. A cover
    # can have textbook texture everywhere else and still fail 2.3 this way.
    B, n_bits = 8, 128
    bw = g.shape[1] // B
    means = np.array([g[(i // bw) * B:(i // bw + 1) * B,
                        (i % bw) * B:(i % bw + 1) * B].mean()
                      for i in range(min(n_bits, (g.shape[0] // B) * bw))])
    room = float(np.mean((means > 8) & (means < 247))) * 100
    report(room >= 90, "embedding headroom",
           f"{room:.0f}% of the first {len(means)} blocks clear of the rails",
           "the strip carrying the payload is saturated; a +/-3 shift clips "
           "under noise and 2.3's robustness result collapses",
           value=room, thr=90, want="min")


def check_cover_smooth(a, path):
    g = _grey(a)
    hf = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
    report(hf < 6.0, "smoothness", f"mean |grad| = {hf:.1f}",
           "needs large flat regions so LSB embedding is VISIBLY worse here",
           value=hf, thr=6.0, want="max")


def check_plt_source(a, path):
    """The critical one. Levels the image does not sample cannot constrain
    the transform, so knots there are unrecoverable and the exercise breaks."""
    g = _grey(a).astype(np.uint8)
    cnt = np.bincount(g.ravel(), minlength=256)
    sup = int((cnt >= 20).sum())
    report(sup >= 240, "level support", f"{sup}/256 levels have >=20 px",
           "knots in unsampled ranges are unrecoverable; pick an image with "
           "both faint detail and bright saturated regions",
           value=sup, thr=240, want="min")
    top = int(np.where(cnt >= 20)[0].max()) if sup else 0
    report(top >= 230, "upper range", f"highest well-sampled level = {top}",
           "no bright pixels: any knot above this is invisible to the fitter",
           value=top, thr=230, want="min")
    dark = 100 * cnt[:32].sum() / g.size
    report(dark < 92, "not pure background", f"{dark:.1f}% of pixels below 32",
           "mostly empty sky; usable but expect a weak upper segment",
           value=dark, thr=92, want="max")


def check_hist_lowcontrast(a, path):
    g = _grey(a)
    rng = g.max() - g.min()
    report(g.std() < 25, "low contrast", f"std = {g.std():.1f}",
           "already contrasty: equalisation will barely change it",
           value=g.std(), thr=25, want="max")
    # Count a level as occupied only if it holds a meaningful share of the
    # image, not if a single stray pixel lands there.
    #
    # This used to test `> 0`. On a multi-megapixel photo that asks "does ANY
    # pixel anywhere have this value", which a handful of hot pixels or JPEG
    # ringing answers yes to across most of the range -- so an image with a
    # textbook-narrow histogram still read as 70% occupied. check_plt_source
    # already uses a >= 20 px floor for exactly this reason; this is the same
    # idea expressed as a fraction so it does not depend on image size.
    cnt = np.bincount(g.astype(np.uint8).ravel(), minlength=256)
    floor = max(1, int(g.size * 1e-4))          # 0.01% of pixels
    occ = 100 * (cnt >= floor).sum() / 256
    report(occ < 60, "narrow occupancy",
           f"{occ:.0f}% of levels hold >={floor} px",
           f"range {rng:.0f} is already wide; HE has little to do",
           value=occ, thr=60, want="max")


def check_hist_natural(a, path):
    """A photographic histogram used for matching or specification.

    It must span enough levels to make inverse-CDF composition meaningful and
    have non-trivial contrast.  This deliberately does not prescribe a shape:
    the source and reference should remain visibly different.
    """
    g = _grey(a).astype(np.uint8)
    cnt = np.bincount(g.ravel(), minlength=256)
    occupied = int((cnt >= max(1, int(g.size * 1e-4))).sum())
    report(g.std() >= 25, "tonal spread", f"std = {g.std():.1f}",
           "too narrow for a general matching/specification source",
           value=g.std(), thr=25, want="min")
    report(occupied >= 150, "level occupancy",
           f"{occupied}/256 levels hold a meaningful number of pixels",
           "too few populated levels: inverse-CDF results will be dominated by ties",
           value=occupied, thr=150, want="min")


def check_hist_clahe(a, path):
    g = _grey(a)
    report(min(g.shape) >= 1024, "size", f"{g.shape[1]}x{g.shape[0]}",
           value=min(g.shape), thr=1024, want="min")
    # Local contrast should vary strongly across the frame, else CLAHE ~ HE.
    ts = max(g.shape) // 8
    stds = [g[y:y+ts, x:x+ts].std()
            for y in range(0, g.shape[0]-ts+1, ts)
            for x in range(0, g.shape[1]-ts+1, ts)]
    cv = np.std(stds) / max(np.mean(stds), 1e-6)
    report(cv > 0.35, "local contrast variation", f"CV of tile std = {cv:.2f}",
           "uniform local contrast: CLAHE will look identical to global HE",
           value=cv, thr=0.35, want="min")


def check_greenscreen(a, path):
    """Estimate fractional-alpha area by keying. Too little and a hard
    threshold scores as well as a proper matte, which kills the problem."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "p5_greenscreen"))
    try:
        from solution import estimate_key_colour, key_soft, to_ycbcr
    except ImportError:
        print("  (needs p5_greenscreen/solution.py)")
        return
    if a.ndim != 3:
        report(False, "colour", "image is greyscale")
        return
    h, w = a.shape[:2]
    report(min(h, w) >= 720, "resolution", f"{w}x{h}",
           "want 1080p+ so the efficiency part has something to chew on",
           value=min(h, w), thr=720, want="min")
    key = estimate_key_colour(a)
    al = key_soft(a, key)
    frac = 100 * np.mean((al > 0.05) & (al < 0.95))
    report(frac >= 3.0, "fractional alpha", f"{frac:.2f}% of pixels",
           "a hard threshold will score as well as a soft matte -- need hair, "
           "motion blur, or translucent material",
           value=frac, thr=3.0, want="min")
    # Backing evenness: a perfectly even backing makes 5.1 trivial.
    yc = to_ycbcr(a)
    d = np.sqrt((yc[..., 1]-key[0])**2 + (yc[..., 2]-key[1])**2)
    # `d < percentile(d, 50)` selects NOTHING when more than half the frame
    # sits at the same chroma distance -- the median then equals the minimum.
    # That is exactly what a flat, heavily compressed backing looks like, so
    # the check was returning NaN on precisely the clips it most needed to
    # judge. Use <= and fall back to the whole frame if it is still degenerate.
    bg = d[d <= np.percentile(d, 50)]
    if bg.size < 2:
        bg = d.ravel()
    sd = float(bg.std())
    report(sd > 3.0, "backing unevenness", f"chroma std = {sd:.1f}",
           "backing is too evenly lit; the naive-RGB failure won't show",
           value=sd, thr=3.0, want="min")


def check_background(a, path):
    h, w = a.shape[:2]
    report(w >= 3200, "pan width", f"{w}px wide",
           "need >=2.5x the frame width to pan across without running out",
           value=w, thr=3200, want="min")
    report(min(h, w) >= 1080, "height", f"{w}x{h}")


ROLES = {
    "conv_base": check_conv_base, "cover_textured": check_cover_textured,
    "cover_smooth": check_cover_smooth, "plt_source": check_plt_source,
    "hist_lowcontrast": check_hist_lowcontrast, "hist_natural": check_hist_natural,
    "hist_clahe": check_hist_clahe,
    "greenscreen": check_greenscreen, "background": check_background,
}

# What the repo currently ships, so --all can audit it in place.
CURRENT = [
    ("conv_base", "p1/base_2048.png"),
    ("cover_textured", "p2/cover_textured.png"),
    ("cover_smooth", "p2/cover_smooth.png"),
    ("plt_source", "p3/field_in.png"),
    ("plt_source", "p3/nebula_in.png"),
    ("hist_lowcontrast", "p4/equalization/input.png"),
    ("hist_natural", "p4/matching/source.png"),
    ("hist_natural", "p4/matching/reference.png"),
    ("hist_natural", "p4/specification/source.png"),
    ("hist_clahe", "p4/local_regions.png"),
    ("greenscreen", "p5/subject_1080p.png"),
    ("background", "p5/background_plate.png"),
]


def centre_crop(a, size):
    """Mirrors build_assets.centre_crop -- largest centred square <= size."""
    h, w = a.shape[:2]
    s = min(size, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return a[y0:y0 + s, x0:x0 + s]


# What build_assets.py crops each raw asset to, so --crop can be inferred.
ROLE_CROP = {"conv_base": 2048, "cover_textured": 1024, "cover_smooth": 1024,
             "plt_source": 1600, "hist_lowcontrast": 2000}


def run(role, path, crop=None):
    print(f"\n{role}  <-  {path}")
    if not os.path.exists(path):
        report(False, "exists", "file not found")
        return
    a = np.asarray(Image.open(path))
    if crop:
        before = a.shape[:2]
        a = centre_crop(a, crop)
        print(f"  (centre-cropped {before[1]}x{before[0]} -> "
              f"{a.shape[1]}x{a.shape[0]}, as build_assets.py will)")
    ROLES[role](a, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=sorted(ROLES))
    ap.add_argument("--all", metavar="DATADIR")
    # Not nargs="?" -- argparse would swallow the following path as its value.
    ap.add_argument("--crop", action="store_true",
                    help="centre-crop to this role's own size before checking, "
                         "as build_assets.py does. Use when auditioning a raw "
                         "download rather than a built asset.")
    ap.add_argument("--crop-size", type=int, metavar="N",
                    help="override the crop size used by --crop")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args()

    if a.all:
        for role, rel in CURRENT:
            run(role, os.path.join(a.all, rel))
    elif a.role:
        crop = a.crop_size or (ROLE_CROP.get(a.role) if a.crop else None)
        if a.crop and crop is None:
            print(f"  (role {a.role} is not cropped by build_assets.py)")
        for p in (a.paths or []):
            for f in sorted(glob.glob(p)) or [p]:
                run(a.role, f, crop=crop)
    else:
        ap.error("give --role ROLE FILE... or --all data/")

    n_bad = VERDICTS.count(False)
    print("\n" + "=" * 56)
    print(f"{len(VERDICTS)-n_bad} passed, {n_bad} failed")
    sys.exit(1 if n_bad else 0)
