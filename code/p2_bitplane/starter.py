"""P2 starter. Implement every function marked TODO."""

import numpy as np


def bit_planes(img):
    """TODO 2.1: (H,W) uint8 -> (8,H,W) uint8 in {0,1}, index 0 = LSB."""
    return np.array([(img >> b) & 1 for b in range(8)], dtype=np.uint8)


def reconstruct(planes, keep):
    """TODO 2.1: rebuild from the `keep` most significant planes."""
    result = np.zeros(planes.shape[1:], dtype=np.uint8)

    for b in range(8 - keep, 8):
        result += planes[b] * (1 << b)

    return result


def gray_encode(img):
    """TODO 2.1: Gray-coded intensities, for the ramp comparison."""
    return img ^ (img >> 1)


def embed_lsb(cover, bits, plane=0):
    """TODO 2.2: write bits into the given plane, raster order."""
    out = cover.copy()

    flat = out.ravel()

    mask = np.uint8(1 << plane)
    flat[: len(bits)] = (flat[: len(bits)] & np.uint8(255 ^ (1 << plane))) | (
        bits * mask
    )

    return out


def extract_lsb(stego, n, plane=0):
    """TODO 2.2: read n bits back out."""
    flat = stego.ravel()

    bits = (flat[:n] >> plane) & 1

    return bits.astype(np.uint8)


# ---- 2.3: block-DCT QIM with repetition coding
#
# Parameters:
#   delta = 48   QIM step. Both codebooks (multiples of 48, and 48k+24) are
#                multiples of the JPEG q75 step for this coefficient, so JPEG
#                requantisation lands back on the lattice instead of between it.
#   coef  = (2,2)  mid-low frequency. Low enough that JPEG keeps it, high
#                enough that changing it is not visible.
#   seed  = 12345  fixes which block carries which bit; the decoder repeats it.
#
# Each of the 128 bits is repeated over 128 blocks scattered across the image
# and recovered by majority vote.

# Orthonormal 8x8 DCT-II matrix, built with NumPy (no scipy).
_k = np.arange(8)
_D = np.cos(np.pi * (2 * _k[None, :] + 1) * _k[:, None] / 16)
_D[0, :] *= np.sqrt(1 / 8)
_D[1:, :] *= np.sqrt(2 / 8)


def _to_blocks(img):
    H, W = img.shape
    return img.reshape(H // 8, 8, W // 8, 8).swapaxes(1, 2)


def _from_blocks(blocks):
    nh, nw = blocks.shape[:2]
    return blocks.swapaxes(1, 2).reshape(nh * 8, nw * 8)


def _bit_of_block(nb, n, seed):
    """Which bit each block carries, under a fixed pseudorandom scatter."""
    order = np.random.default_rng(seed).permutation(nb)
    assign = np.empty(nb, dtype=np.int64)
    assign[order] = np.arange(nb) % n
    return assign


def embed_robust(cover, bits, delta=48.0, coef=(2, 2), seed=12345):
    """Quantisation index modulation on one DCT coefficient per 8x8 block."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = len(bits)
    u, v = coef

    blocks = _to_blocks(cover.astype(np.float64))
    nh, nw = blocks.shape[:2]
    nb = nh * nw

    C = (_D @ blocks @ _D.T).reshape(nb, 8, 8)

    # Bit 0 -> lattice {k*delta}, bit 1 -> lattice {k*delta + delta/2}
    carried = bits[_bit_of_block(nb, n, seed)]
    offset = np.where(carried == 1, delta / 2, 0.0)

    c = C[:, u, v]
    C[:, u, v] = delta * np.round((c - offset) / delta) + offset

    rec = _D.T @ C.reshape(nh, nw, 8, 8) @ _D

    return np.clip(np.round(_from_blocks(rec)), 0, 255).astype(np.uint8)


def extract_robust(stego, cover, n, delta=48.0, coef=(2, 2), seed=12345):
    """Blind decoder: `cover` is accepted to match the starter signature but is
    never read. QIM decides from the stego coefficient alone."""
    u, v = coef

    blocks = _to_blocks(stego.astype(np.float64))
    nh, nw = blocks.shape[:2]
    nb = nh * nw

    C = (_D @ blocks @ _D.T).reshape(nb, 8, 8)
    c = C[:, u, v]

    # Nearer lattice wins
    res = np.mod(c, delta)
    d0 = np.minimum(res, delta - res)
    d1 = np.abs(res - delta / 2)
    hard = (d1 < d0).astype(np.int64)

    # Majority vote over the blocks carrying each bit
    assign = _bit_of_block(nb, n, seed)
    votes = np.bincount(assign, weights=hard, minlength=n)
    counts = np.bincount(assign, minlength=n)

    return (votes > counts / 2).astype(np.uint8)


# ---- provided channels: do not modify, these are what you are graded against
def degrade_gaussian(img, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(
        np.round(img.astype(np.float64) + rng.normal(0, sigma, img.shape)), 0, 255
    ).astype(np.uint8)


def degrade_jpeg(img, quality):
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def ber(a, b):
    return float(np.mean(a != b))


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


if __name__ == "__main__":
    from PIL import Image

    # Load stego image
    stego = np.asarray(Image.open("../../images/p2/decode_me.png").convert("L")).astype(
        np.uint8
    )

    # Extract all LSBs
    bits = extract_lsb(stego, stego.size)

    # Check sync marker
    sync = bits[:8]

    print("Sync marker:", sync)
    print("Expected    :", np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8))
