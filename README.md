# Digital Image Processing — Assignment 1

This is the student release. It contains the assignment source, every supplied
P1–P4 image, starter code, the fixed test-channel functions, and image credits.

## Start here

1. Read or compile `assignment.tex`.
2. Install the Python dependencies with:

   ```text
   python -m pip install -r requirements.txt
   ```

3. Implement the `TODO` functions in the relevant `code/*/starter.py` files.
   You may rename or copy your completed files for submission.
4. Run programs from this folder, so paths such as
   `images/p2/decode_me.png` resolve exactly as written in the handout.

The starter files are scaffolds, not an autograder. Your report and viva must
explain your own implementation and measurements.

## Supplied data map

| Problem | Supplied files |
|---|---|
| P1 | `images/p1/base_2048.png` |
| P2 | `images/p2/cover_textured.png`, `cover_smooth.png`, `decode_me.png` |
| P3 A | `images/p3/field_in.png`, `field_out_A.png` |
| P3 B | `images/p3/field_in.png`, `field_out_B.png` |
| P3 C | `images/p3/nebula_in.png`, `nebula_out_C.png` |
| P3 D | `images/p3/field_skycrop_in.png`, `field_skycrop_out_D.png` |
| P4.1 | `images/p4/equalization/input.png` |
| P4.2(a) | `images/p4/matching/source.png`, `reference.png` |
| P4.2(b) | `images/p4/specification/source.png` |
| P4.2(c) | `images/p4/colour_source.png` |
| P4.3 | `images/p4/local_regions.png` |
| P4.4 | `images/p4/exposure/ev0.png` through `ev4.png` |

P2's clean cover for `decode_me.png` is `cover_textured.png`.

P5 images are deliberately not supplied. Source your own qualifying green-screen
footage and background plate as required by the handout. The optional helper
`code/assets/ingest_greenscreen.py` can extract and inspect 48 frames. It must be
run from the bundle root:

```text
python code/assets/ingest_greenscreen.py path/to/clip.mp4 --frames 48 --report
```

This helper uses `code/assets/check_asset.py` and writes selected frames under
`images/p5/`. You remain responsible for visually checking that your footage
contains real hair, motion blur, or translucent detail.

## Files intentionally not supplied

- A P2 ramp: construct it with NumPy as requested.
- P5 footage or background: source both yourself.
- Processed outputs, reference answers, solution code, hidden-answer arrays, or
  instructor grading material.

See `ATTRIBUTION.md` for licences and source credits for the supplied images.
`MANIFEST_SHA256.txt` records SHA-256 hashes for the release files so accidental
asset corruption can be detected after download.
