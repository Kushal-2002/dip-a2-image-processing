"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""

import time
import numpy as np
from scipy.signal import convolve2d, correlate2d  # checking only
import matplotlib.pyplot as plt


def kernel_bank(k=15):
    """Provided. Do not modify -- your rank table must match these kernels."""
    ax = np.arange(k) - (k - 1) / 2
    box = np.ones((k, k)) / (k * k)
    s = k / 6.0
    g1 = np.exp(-(ax**2) / (2 * s * s))
    g1 /= g1.sum()
    gauss = np.outer(g1, g1)
    sobel = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    log = (r2 - 2 * s * s) / (s**4) * np.exp(-r2 / (2 * s * s))
    motion0 = np.zeros((k, k))
    motion0[k // 2, :] = 1.0 / k
    disk = (r2 <= (k / 2.0) ** 2).astype(float)
    disk /= disk.sum()
    rand = np.random.default_rng(0).normal(size=(k, k))
    rand /= np.abs(rand).sum()
    return {
        "box": box,
        "gaussian": gauss,
        "sobel3": sobel,
        "log": log,
        "log_dc_removed": log - log.mean(),
        "motion_0deg": motion0,
        "motion_45deg": np.eye(k) / k,
        "disk": disk,
        "random": rand,
    }


def numeric_rank(K, tol=1e-10):
    S = np.linalg.svd(K, compute_uv=False)
    return np.sum(S > tol)


def conv2d_loops(img, K):
    H, W = img.shape
    kh, kw = K.shape

    # True convolution: flip kernel 180 degrees
    K_flip = np.flip(K)

    # Zero padding
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    # Same-size output
    out = np.zeros((H, W), dtype=np.float64)

    # Four nested loops
    for i in range(H):
        for j in range(W):
            for ki in range(kh):
                for kj in range(kw):
                    out[i, j] += padded[i + ki, j + kj] * K_flip[ki, kj]

    return out


def conv2d_taps(img, K):
    H, W = img.shape
    kh, kw = K.shape

    # True convolution
    K_flip = np.flip(K)

    # Zero padding
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    out = np.zeros((H, W), dtype=np.float64)

    # Loop over kernel taps
    for ki in range(kh):
        for kj in range(kw):
            out += K_flip[ki, kj] * padded[ki : ki + H, kj : kj + W]

    return out


def conv2d_im2col(img, K):
    H, W = img.shape
    kh, kw = K.shape

    # True convolution
    K_flip = np.flip(K)

    # Zero padding
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    # Extract every image patch
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))

    # Convert patches into rows
    patches = windows.reshape(H * W, kh * kw)

    # Flatten flipped kernel
    kernel_vector = K_flip.reshape(kh * kw)

    # ONE matrix multiplication
    out = patches @ kernel_vector

    # Put back into image shape
    return out.reshape(H, W)


def conv2d_fft(img, K):
    H, W = img.shape
    kh, kw = K.shape

    # No flip here: multiplying the FFTs already IS convolution, so flipping
    # the kernel first would give correlation instead.

    # Size of full convolution
    fh = H + kh - 1
    fw = W + kw - 1

    # Zero-padded arrays
    img_pad = np.zeros((fh, fw), dtype=np.float64)
    K_pad = np.zeros((fh, fw), dtype=np.float64)

    img_pad[:H, :W] = img
    K_pad[:kh, :kw] = K

    # FFT
    F_img = np.fft.fft2(img_pad)
    F_K = np.fft.fft2(K_pad)

    # Convolution in frequency domain
    F_result = F_img * F_K

    # Back to spatial domain
    full = np.fft.ifft2(F_result).real

    # Extract 'same' region
    pad_h = kh // 2
    pad_w = kw // 2

    return full[pad_h : pad_h + H, pad_w : pad_w + W]


def conv2d_separable(img, K, tol=1e-10):
    # SVD of the kernel
    U, S, Vt = np.linalg.svd(K)

    # Check that kernel is rank-1
    rank = numeric_rank(K, tol)

    if rank != 1:
        raise ValueError("Kernel is not rank-1")

    # Create the two 1D filters
    h = np.sqrt(S[0]) * U[:, 0]
    v = np.sqrt(S[0]) * Vt[0, :]

    # K[i, j] = h[i] * v[j], so h runs down the rows (column filter) and v
    # runs across the columns (row filter).
    temp = conv2d_taps(img, h.reshape(-1, 1))
    out = conv2d_taps(temp, v.reshape(1, -1))

    return out


def conv2d_lowrank(img, K, r):
    U, S, Vt = np.linalg.svd(K)

    # Make sure r is valid
    r = min(r, len(S))

    # Output starts at zero
    out = np.zeros_like(img, dtype=np.float64)

    # Add r separable components
    for i in range(r):
        h = np.sqrt(S[i]) * U[:, i]
        v = np.sqrt(S[i]) * Vt[i, :]

        # Vertical convolution (h runs down the rows)
        temp = conv2d_taps(img, h.reshape(-1, 1))

        # Horizontal convolution (v runs across the columns)
        component = conv2d_taps(temp, v.reshape(1, -1))

        # Add this component
        out += component

    return out


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from scipy.signal import convolve2d  # checking only

    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[
        :128, :128
    ]
    K = kernel_bank(7)["gaussian"]
    ref = convolve2d(img, K, mode="same", boundary="fill")

    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        try:
            print(f"{fn.__name__:16s} max|err| = {np.abs(fn(img, K) - ref).max():.3e}")
        except NotImplementedError:
            print(f"{fn.__name__:16s} not implemented")

    # Asymmetric kernel test
    K_asym = np.array([[1, 2, 0], [3, 4, 5], [0, 6, 7]], dtype=float)

    conv = conv2d_loops(img, K_asym)
    corr = correlate2d(img, K_asym, mode="same", boundary="fill")

    print("max |convolution - correlation| =", np.max(np.abs(conv - corr)))

    # Runtime test
    K_runtime = kernel_bank(7)["gaussian"]

    start = time.perf_counter()
    conv2d_loops(img, K_runtime)
    end = time.perf_counter()

    elapsed = end - start

    print("128x128, k=7 runtime:", elapsed, "seconds")

    # Extrapolate to 2048x2048, k=15. The loop count is H*W*k*k, so the cost
    # scales as N^2 * k^2.
    scale = (2048 / 128) ** 2 * (15 / 7) ** 2
    predicted = elapsed * scale

    print("scale factor to 2048x2048, k=15:", scale)
    print("predicted 2048x2048, k=15 runtime:", predicted, "seconds")
    print("  =", predicted / 60, "minutes")

    # Separable convolution test
    K_sep = kernel_bank(15)["gaussian"]

    sep = conv2d_separable(img, K_sep)
    ref_sep = convolve2d(img, K_sep, mode="same", boundary="fill")

    print("separable max|err| =", np.max(np.abs(sep - ref_sep)))

    # Numerical rank test
    kernels = kernel_bank(15)

    for name, K in kernels.items():
        print(name, "rank =", numeric_rank(K))

    # Low-rank test
    K_lr = kernel_bank(15)["disk"]

    lr = conv2d_lowrank(img, K_lr, 5)
    ref_lr = convolve2d(img, K_lr, mode="same", boundary="fill")

    print("lowrank r=5 max|err| =", np.max(np.abs(lr - ref_lr)))

    img512 = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[
        :512, :512
    ]

    kernels21 = kernel_bank(21)

    psnr_results = {}

    for name in ["disk", "log", "motion_45deg", "random"]:
        K = kernels21[name]

        print("\nKernel:", name)
        print("Rank:", numeric_rank(K))

        exact = conv2d_taps(img512, K)

        scores = []

        for r in range(1, 22):
            approx = conv2d_lowrank(img512, K, r)
            score = psnr(approx, exact)

            scores.append(score)
            print("r =", r, "PSNR =", score)

        psnr_results[name] = scores

    plt.figure(figsize=(8, 5))

    for name, scores in psnr_results.items():
        plt.plot(range(1, 22), scores, marker="o", label=name)

    plt.xlabel("Rank r")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Rank of Low-Rank Approximation")
    plt.legend()
    plt.grid(True)

    plt.show()

    print("\nNormalized singular-value spectra:")

    for name, K in kernels.items():
        S = np.linalg.svd(K, compute_uv=False)

        S_norm = S / S[0]

        print("\n", name)
        print(S_norm)

    plt.figure(figsize=(8, 5))

    for name, K in kernels.items():
        S = np.linalg.svd(K, compute_uv=False)
        S_norm = S / S[0]

        plt.semilogy(range(1, len(S_norm) + 1), S_norm, marker="o", label=name)

    plt.xlabel("Singular value index")
    plt.ylabel("Normalized singular value")
    plt.title("Normalized Singular-Value Spectrum")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ============================================================
    # EXPERIMENT A: Runtime vs Kernel Size
    # ============================================================

    print("\n===== Experiment A: Runtime vs Kernel Size =====")

    # Fixed 512 x 512 image
    img512 = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[
        :512, :512
    ]

    # Kernel sizes to test
    k_values = [3, 7, 11, 15, 21, 31]

    # Store runtimes
    loops_times = []
    taps_times = []
    im2col_times = []
    fft_times = []

    for k in k_values:
        print("\nKernel size k =", k)

        # Create Gaussian kernel
        K = kernel_bank(k)["gaussian"]

        # -------------------------
        # 1. Four nested loops
        # -------------------------
        start = time.perf_counter()

        conv2d_loops(img512, K)

        elapsed = time.perf_counter() - start
        loops_times.append(elapsed)

        print("loops :", elapsed, "seconds")

        # -------------------------
        # 2. Taps
        # -------------------------
        start = time.perf_counter()

        conv2d_taps(img512, K)

        elapsed = time.perf_counter() - start
        taps_times.append(elapsed)

        print("taps  :", elapsed, "seconds")

        # -------------------------
        # 3. Im2col
        # -------------------------
        start = time.perf_counter()

        conv2d_im2col(img512, K)

        elapsed = time.perf_counter() - start
        im2col_times.append(elapsed)

        print("im2col:", elapsed, "seconds")

        # -------------------------
        # 4. FFT
        # -------------------------
        start = time.perf_counter()

        conv2d_fft(img512, K)

        elapsed = time.perf_counter() - start
        fft_times.append(elapsed)

        print("fft   :", elapsed, "seconds")

    # ============================================================
    # Print summary table
    # ============================================================

    print("\n===== Summary =====")

    print(f"{'k':>5} {'loops':>12} {'taps':>12} {'im2col':>12} {'fft':>12}")

    for i, k in enumerate(k_values):
        print(
            f"{k:5d} "
            f"{loops_times[i]:12.6f} "
            f"{taps_times[i]:12.6f} "
            f"{im2col_times[i]:12.6f} "
            f"{fft_times[i]:12.6f}"
        )

    # ============================================================
    # Fitted slopes on the log-log data
    # ============================================================

    print("\n===== Fitted slopes (runtime ~ k^p) =====")

    slopes_k = {}

    for label, times in [
        ("loops", loops_times),
        ("taps", taps_times),
        ("im2col", im2col_times),
        ("fft", fft_times),
    ]:
        slope = np.polyfit(np.log(k_values), np.log(times), 1)[0]
        slopes_k[label] = slope

        print(f"{label:8s} p = {slope:.2f}")

    # ============================================================
    # Plot
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.loglog(k_values, loops_times, marker="o", label=f"loops (p={slopes_k['loops']:.2f})")
    plt.loglog(k_values, taps_times, marker="o", label=f"taps (p={slopes_k['taps']:.2f})")
    plt.loglog(k_values, im2col_times, marker="o", label=f"im2col (p={slopes_k['im2col']:.2f})")
    plt.loglog(k_values, fft_times, marker="o", label=f"fft (p={slopes_k['fft']:.2f})")

    plt.xlabel("Kernel size k")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Kernel Size (512x512)")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ============================================================
    # EXPERIMENT B: Runtime vs Image Size at k = 15
    # ============================================================

    print("\n===== Experiment B: Runtime vs Image Size (k=15) =====")

    base = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)

    N_values = [128, 256, 512, 1024, 2048]
    K15 = kernel_bank(15)["gaussian"]

    # im2col builds an (N*N, k*k) float64 patch matrix. Skip it when that
    # would not fit comfortably in RAM -- see the memory table below.
    MEM_LIMIT_GB = 4.0

    taps_N = []
    im2col_N = []
    im2col_N_values = []
    fft_N = []
    mem_N = []

    for N in N_values:
        print("\nImage size N =", N)

        imgN = base[:N, :N]

        # Patch-matrix memory: N * N rows, k * k columns, 8 bytes per float64
        mem_gb = N * N * 15 * 15 * 8 / 1e9
        mem_N.append(mem_gb)

        start = time.perf_counter()
        conv2d_taps(imgN, K15)
        elapsed = time.perf_counter() - start
        taps_N.append(elapsed)
        print("taps  :", elapsed, "seconds")

        if mem_gb <= MEM_LIMIT_GB:
            start = time.perf_counter()
            conv2d_im2col(imgN, K15)
            elapsed = time.perf_counter() - start
            im2col_N.append(elapsed)
            im2col_N_values.append(N)
            print("im2col:", elapsed, "seconds")
        else:
            print(f"im2col: SKIPPED, patch matrix needs {mem_gb:.2f} GB")

        start = time.perf_counter()
        conv2d_fft(imgN, K15)
        elapsed = time.perf_counter() - start
        fft_N.append(elapsed)
        print("fft   :", elapsed, "seconds")

    # ============================================================
    # im2col peak memory table
    # ============================================================

    print("\n===== im2col patch-matrix memory (k=15, float64) =====")

    print(f"{'N':>6} {'patch matrix':>20} {'memory (GB)':>14}")

    for i, N in enumerate(N_values):
        print(f"{N:6d} {f'{N * N} x {15 * 15}':>20} {mem_N[i]:14.3f}")

    # ============================================================
    # Fitted slopes for Experiment B
    # ============================================================

    print("\n===== Fitted slopes (runtime ~ N^p) =====")

    slope_taps_N = np.polyfit(np.log(N_values), np.log(taps_N), 1)[0]
    slope_fft_N = np.polyfit(np.log(N_values), np.log(fft_N), 1)[0]
    slope_im2col_N = np.polyfit(np.log(im2col_N_values), np.log(im2col_N), 1)[0]

    print(f"taps   p = {slope_taps_N:.2f}")
    print(f"im2col p = {slope_im2col_N:.2f}")
    print(f"fft    p = {slope_fft_N:.2f}")

    plt.figure(figsize=(8, 5))

    plt.loglog(N_values, taps_N, marker="o", label=f"taps (p={slope_taps_N:.2f})")
    plt.loglog(im2col_N_values, im2col_N, marker="o", label=f"im2col (p={slope_im2col_N:.2f})")
    plt.loglog(N_values, fft_N, marker="o", label=f"fft (p={slope_fft_N:.2f})")

    plt.xlabel("Image size N")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Image Size (k=15)")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ============================================================
    # 1.4(c): separable speedup at k = 15
    # ============================================================

    print("\n===== Separable speedup at k=15 (theory: k/2 = 7.5x) =====")

    for N in [512, 2048]:
        imgN = base[:N, :N]

        start = time.perf_counter()
        conv2d_taps(imgN, K15)
        t_taps = time.perf_counter() - start

        start = time.perf_counter()
        conv2d_separable(imgN, K15)
        t_sep = time.perf_counter() - start

        print(
            f"N={N:5d}  taps={t_taps:8.4f}s  separable={t_sep:8.4f}s  "
            f"speedup={t_taps / t_sep:5.2f}x"
        )
