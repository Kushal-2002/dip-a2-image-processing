"""Turn a downloaded green screen clip into the P5 frame sequence.

    python assets/ingest_greenscreen.py CLIP.mp4
    python assets/ingest_greenscreen.py CLIP.mp4 --frames 48 --report

Commons has no usable chroma-key footage, so this is the one asset that has to
be sourced by hand:

    https://pixabay.com/videos/search/green%20screen%20person/
    https://mixkit.co/free-stock-video/green-screen/

Pick a clip with **loose hair, fur or sheer fabric**. That is not a stylistic
preference: problem 5.2 asks students to show a soft matte beating a hard
threshold, and the only thing that makes a soft matte win is genuinely
fractional alpha. A clean-edged subject against a flat backing can be keyed
just as well by a single threshold, and 5.2 collapses into a tautology.

What this does
--------------
1. Decodes the clip with the ffmpeg that ships inside imageio-ffmpeg, so
   there is nothing to install.
2. Slides a window over the clip and scores every candidate run of `--frames`
   frames on the check_asset.py `greenscreen` role -- fractional alpha and
   backing unevenness. Clips often open on an empty stage and only bring the
   subject in later; this finds the part worth keeping.
3. Writes the winning window to images/p5/frames/ at 1280x720, and the middle
   frame to images/p5/subject_1080p.png and subject_480p.png.
4. With --report, writes a diagnostic PNG so you can SEE whether the hair
   survived keying, which no single number tells you reliably.

Ground truth
------------
The synthetic generator in build_assets.py produces a matte it knows exactly.
Real footage does not come with one, so `_solution_alpha.png` stays synthetic
and instructor/verify.py keeps using the synthetic pair for the three checks
that need ground truth. Students key the real footage; the reference numbers
are still measured against something exactly known.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "p5_greenscreen"))
import check_asset  # noqa: E402


def data_dir():
    env = os.environ.get("DIP_DATA")
    if env:
        return os.path.abspath(env)
    bundled = os.path.join(os.path.dirname(ROOT), "images")
    return bundled if os.path.isdir(bundled) else os.path.join(ROOT, "data")


DATA = data_dir()
W, H = 1280, 720


def load_solution():
    """The reference keyer, if this is the instructor copy.

    Students get this script without p5_greenscreen/solution.py, so scoring
    and the diagnostic are unavailable to them -- the script still does the
    useful part, which is decoding to a clean 48-frame sequence at a sane
    resolution. There is no autograder; choosing good footage is a judgement
    call the viva examines, not something a number decides.
    """
    try:
        from solution import estimate_key_colour, key_soft
        return estimate_key_colour, key_soft
    except ImportError:
        return None, None


HAVE_SOLUTION = load_solution()[0] is not None


def decode(path, max_frames=600):
    """Frames as uint8 RGB at 1280x720, using imageio-ffmpeg's bundled binary."""
    import imageio.v2 as imageio
    try:
        rdr = imageio.get_reader(path, "ffmpeg")
    except Exception as e:
        sys.exit(f"could not open {path}: {e}\n"
                 "If this is not a video, pass a directory of frames instead.")
    out = []
    for i, fr in enumerate(rdr):
        if i >= max_frames:
            break
        im = Image.fromarray(fr).convert("RGB")
        if im.size[0] > W or im.size[1] > H:
            im.thumbnail((W, H), Image.LANCZOS)   # downscale only
        out.append(np.asarray(im))
    rdr.close()
    if not out:
        sys.exit("decoded zero frames")
    return out


def load_dir(path):
    files = sorted(glob.glob(os.path.join(path, "*.png")) +
                   glob.glob(os.path.join(path, "*.jpg")))
    if not files:
        sys.exit(f"no .png/.jpg frames in {path}")
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        out.append(np.asarray(im))
    return out


def score(frame):
    """(fractional alpha %, backing chroma std) for one frame."""
    ok, results, _ = check_asset.evaluate("greenscreen", frame)
    vals = {}
    for r in results:
        if r[1] == "fractional alpha":
            vals["frac"] = float(r[2].split("%")[0])
        elif r[1] == "backing unevenness":
            vals["backing"] = float(r[2].split("=")[1])
    return vals.get("frac", 0.0), vals.get("backing", 0.0)


def pick_window(frames, n, stride):
    """Best contiguous run of n frames, scored on a sample of each window."""
    if len(frames) <= n:
        return 0, *score(frames[len(frames) // 2])
    best = (0, -np.inf, 0.0, 0.0)
    for s in range(0, len(frames) - n + 1, stride):
        probe = [frames[s], frames[s + n // 2], frames[s + n - 1]]
        fr = float(np.nanmean([score(p)[0] for p in probe]))
        bk = float(np.nanmean([score(p)[1] for p in probe]))
        fr = 0.0 if np.isnan(fr) else fr
        bk = 0.0 if np.isnan(bk) else bk
        # fractional alpha is what 5.2 lives on; backing unevenness is a
        # secondary tiebreak for 5.1.
        val = fr + 0.2 * bk
        if val > best[1]:
            best = (s, val, fr, bk)
    return best[0], best[2], best[3]


def diagnostic(frame, path):
    """Frame, soft matte, and a zoom on the top of the subject.

    The zoom is the point: strand-level detail either survives keying or it
    does not, and that is obvious by eye and slippery to measure. Two attempts
    at an automatic hair score both mis-ranked a deliberately blurred control
    above genuine strands, so this reports the picture and lets you judge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    est, soft = load_solution()
    key = est(frame)
    a = soft(frame, key)
    ys, xs = np.where(a > 0.5)
    if len(ys):
        y0 = max(0, ys.min() - 10); x0 = max(0, int(np.median(xs)) - 160)
        crop = (slice(y0, y0 + 260), slice(x0, x0 + 320))
    else:
        crop = (slice(0, 260), slice(0, 320))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].imshow(frame); ax[0, 0].set_title("source frame", fontsize=9)
    ax[0, 1].imshow(a, cmap="gray", vmin=0, vmax=1)
    ax[0, 1].set_title("soft matte (white = opaque)", fontsize=9)
    ax[1, 0].imshow(frame[crop]); ax[1, 0].set_title("zoom: top of subject", fontsize=9)
    ax[1, 1].imshow(a[crop], cmap="gray", vmin=0, vmax=1)
    ax[1, 1].set_title("zoom: matte — do individual strands survive?", fontsize=9)
    for r in ax.ravel():
        r.axis("off")
    frac = 100 * np.mean((a > 0.05) & (a < 0.95))
    fig.suptitle(f"green screen ingest — fractional alpha {frac:.2f}% "
                 f"(need >= 3%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clip", help="video file, or a directory of frames")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--stride", type=int, default=8,
                    help="window step when searching the clip")
    ap.add_argument("--max-decode", type=int, default=600)
    ap.add_argument("--report", action="store_true",
                    help="write a diagnostic PNG next to the frames")
    ap.add_argument("--dry-run", action="store_true",
                    help="score the clip, write nothing")
    a = ap.parse_args()

    frames = (load_dir(a.clip) if os.path.isdir(a.clip)
              else decode(a.clip, a.max_decode))
    h0, w0 = frames[0].shape[:2]
    print(f"decoded {len(frames)} frames at {w0}x{h0}")
    if min(w0, h0) < 720:
        print(f"  WARNING: {w0}x{h0} is below the 720px floor the greenscreen "
              f"role asks for.\n"
              "  Not upscaling -- interpolation invents pixels, it does not "
              "recover strand detail,\n"
              "  and the efficiency part of 5.4 becomes meaningless at this "
              "size.")

    if not HAVE_SOLUTION:
        print("  (no reference keyer available -- taking the middle window and "
              "skipping\n   the scoring pass; judge the footage by eye against "
              "the checklist in the handout)")
        start = max(0, (len(frames) - a.frames) // 2)
        frac = backing = float("nan")
    else:
        start, frac, backing = pick_window(frames, a.frames, a.stride)
    print(f"best window: frames {start}..{start + a.frames - 1}")
    if not HAVE_SOLUTION:
        print("  fractional alpha: not measured")
    else:
        print(f"  fractional alpha {frac:.2f}%  (want >= 3.0)")
        print(f"  backing chroma std {backing:.1f}  (want > 3.0 for 5.1 to bite)")
    if HAVE_SOLUTION and frac < 3.0:
        print("  WARNING: below the fractional-alpha floor. A hard threshold "
              "will score as well as a soft matte and 5.2 loses its point.\n"
              "  Pick a clip with looser hair, sheer fabric, or more motion "
              "blur.")
    if HAVE_SOLUTION and backing <= 3.0:
        print("  NOTE: the backing is very evenly lit, so 5.1's "
              "naive-RGB failure will be undramatic. Usable, but consider "
              "dropping 5.1's 'unevenly lit' wording.")

    if a.dry_run:
        return

    fdir = os.path.join(DATA, "p5", "frames")
    if os.path.isdir(fdir):
        shutil.rmtree(fdir)
    os.makedirs(fdir, exist_ok=True)
    keep = frames[start:start + a.frames]
    for i, f in enumerate(keep):
        Image.fromarray(f).save(os.path.join(fdir, f"f{i:03d}.png"))
    # Tell build_assets.py to stop regenerating synthetic frames over the top
    # of these. Without the marker the next rebuild silently destroys the clip.
    with open(os.path.join(fdir, "_REAL_FOOTAGE"), "w", encoding="utf-8") as fh:
        fh.write(f"source: {os.path.abspath(a.clip)}\n"
                 f"window: frames {start}..{start + a.frames - 1}\n"
                 f"fractional alpha: {frac:.2f}%\n"
                 f"backing chroma std: {backing:.1f}\n"
                 "Delete this file to let build_assets.py regenerate synthetic "
                 "frames again.\n")
    print(f"wrote {len(keep)} frames -> {fdir}")

    mid = keep[len(keep) // 2]
    Image.fromarray(mid).save(os.path.join(DATA, "p5", "subject_1080p.png"))
    Image.fromarray(mid).resize((854, 480), Image.LANCZOS).save(
        os.path.join(DATA, "p5", "subject_480p.png"))
    print("wrote subject_1080p.png and subject_480p.png from the middle frame")
    print("NOTE: _solution_alpha.png is left alone -- it is the synthetic "
          "ground truth\n      instructor/verify.py measures against, and real "
          "footage has none.")

    if a.report and not HAVE_SOLUTION:
        print("--report needs the reference keyer; skipped")
    elif a.report:
        rp = os.path.join(DATA, "p5", "_ingest_report.png")
        diagnostic(mid, rp)
        print(f"wrote {rp} -- open it and check the bottom-right panel")

    print("\nNext:")
    print("  python assets/check_asset.py --role greenscreen "
          f"{os.path.join('..', os.path.basename(DATA), 'p5', 'frames', 'f000.png')}")
    print("  python instructor/verify.py")


if __name__ == "__main__":
    main()
