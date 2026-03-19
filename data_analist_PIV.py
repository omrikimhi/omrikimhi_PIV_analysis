#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIV droplet concentration analysis
----------------------------------

Experiment context:
- PIV images of a falling “concentration droplet” in a water tank.
- Each index has TWO closely-spaced images: LA and LB (separation dt1_ns).
- Consecutive indices are separated by dt2_s.
- We update the cloud boundary every dt3 = X * dt2_s (every X indices).

Processing logic: what the code do?
1) Pair parsing (LA/LB): scan RAW_DIR, find complete LA/LB pairs, build idxs.
2) Background removal: estimate a static background from the first N_BG_FRAMES
   (median over time), subtract it, clamp negatives to 0 (“absolute black”).
3) Preprocess intensity to I_norm in [0,1]:
   - median filter suppresses small bright tracer particles
   - robust percentile normalization reduces sensitivity to lighting changes
   - mild Gaussian smoothing stabilizes boundary extraction
4) Boundary = OUTER ENVELOPE of the cloud as a GENERAL contour:
   - threshold at a LOW onset value THRESH_FIXED (“cloud almost vanished”)
   - remove tiny islands + morphological closing to fill tiny gaps
   - keep only the largest connected component (the cloud)
   - extract the outer boundary with find_contours; choose the longest contour
5) “Moving boundary”: re-detect the contour every X frames so it follows the
   falling droplet (reduces sensitivity to pure translation when interpreting trends).
6) Diffusive-flux proxy through the contour:
      Flux_proxy ≈ ∫ ( -∇C · n̂ ) ds
   with a synthetic mapping C = I_norm * C_MAX (until we have calibration).
   If we know a diffusion coefficient D, set D_DIFF to compute ∫(-D∇C·n̂)ds.
7) Outputs: animation (image + contour + flux time series) and optional GIF saved
   OUTSIDE RawData (in BASE_DIR or an output folder).

Notes
- Without calibration (and without D), treat flux as a trend/proxy, not an absolute value.
"""

# =========================
# Imports
# =========================

from pathlib import Path
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dotenv import find_dotenv, load_dotenv

from skimage import io
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import find_contours
from skimage.measure import label  # connected components

from scipy.ndimage import median_filter


# =========================
# SETTINGS
# =========================

BASE_DIR = Path(__file__).resolve().parent   # התיקייה של data_analist_PIV.py


def resolve_raw_dir(base_dir):
    load_dotenv(find_dotenv(usecwd=True))

    raw_dir_value = os.getenv("PIV_RAW_DIR")
    if raw_dir_value:
        raw_dir = Path(raw_dir_value).expanduser()
        if not raw_dir.is_absolute():
            raw_dir = base_dir / raw_dir
        return raw_dir.resolve()

    return (base_dir / "RawData").resolve()


RAW_DIR = resolve_raw_dir(BASE_DIR)

if not RAW_DIR.exists():
    raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")

# Timing
dt1_ns = 4000.0          # within-PIV-pair separation [ns] 
dt2_s  = 0.05            # time between indices [seconds] 
X      = 5              # update boundary every X frames  <-- dt3 = X * dt2

# Spatial scaling
PX_PER_MM = 16.0
M_PER_PX  = 1e-3 / PX_PER_MM   # meters per pixel

# Synthetic intensity -> concentration mapping (placeholder calibration)
C_MAX = 1.0              # [kg/m^3] when I_norm = 1

# Background estimation
N_BG_FRAMES = 10         # number of early frames used to estimate background
BG_METHOD   = "median"   # options: mean (if there is frames without drop) or median (if the drop is in the initial frames)

# Preprocessing to suppress tracer particles
MEDIAN_SIZE = 3          # 3 or 5; higher = stronger speckle removal but more blur
SMOOTH_SIGMA = 1.0       # mild Gaussian smoothing for boundary stability

# Outer-envelope boundary detection
# For "outer envelope", FIXED low threshold is usually better than Otsu.
THRESH_METHOD = "fixed"  # "fixed" or "otsu" for autometic disition
THRESH_FIXED  = 0.05     # brightness percentage for the border
MIN_OBJ_PIXELS = 2000    # remove tiny islands (increase if boundary still noisy)
CLOSING_RADIUS = 2       # morphological closing radius (pixels)

# Flux calculation
D_DIFF = None            # diffusion coefficient [m^2/s] (None -> proxy without D)
BAND_PX = 3              # sampling band (pixels) for gradient evaluation is implicit here

# Output / animation
FPS = 10                 # gif fps if saving
SAVE_GIF = False         # set True to save animation as GIF
GIF_NAME = "flux_boundary_animation.gif"
ANALYSIS_DIR = BASE_DIR / "Analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


# =========================
# Helpers: file parsing
# =========================

def list_pairs(raw_dir):
    """
    Scan RAW_DIR for .tif/.tiff files and group them into LA/LB pairs by index.
    Expected filename contains:
      - ".LA." or ".LB."
      - an index like "_001200." somewhere in the name
    """
    files = [p for p in raw_dir.iterdir()
             if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]

    pairs = {}  # idx -> {"LA": Path, "LB": Path}

    for f in sorted(files):
        nameU = f.name.upper()

        # Identify LA/LB
        if ".LA." in nameU:
            ab = "LA"
        elif ".LB." in nameU:
            ab = "LB"
        else:
            continue

        # Extract 6-digit index after underscore and before dot: _001200.
        m = re.search(r"_(\d{6})\.", f.name)
        if not m:
            continue

        idx = int(m.group(1))
        pairs.setdefault(idx, {})[ab] = f

    idxs = sorted([k for k, v in pairs.items() if "LA" in v and "LB" in v])
    return idxs, pairs


def read_pair_mean(pairs, idx):
    """Read LA and LB and return their mean as float32."""
    A = io.imread(pairs[idx]["LA"]).astype(np.float32)
    B = io.imread(pairs[idx]["LB"]).astype(np.float32)
    return 0.5 * (A + B)


# =========================
# Background + preprocessing
# =========================

def estimate_background(idxs, pairs, n_frames=10, method="median"):
    """Estimate static background from the first n_frames (mean of LA/LB each)."""
    n = min(n_frames, len(idxs))
    stack = []
    for i in range(n):
        stack.append(read_pair_mean(pairs, idxs[i]))
    stack = np.stack(stack, axis=0)  # (n, H, W)

    if method == "median":
        bg = np.median(stack, axis=0)
    else:
        bg = np.mean(stack, axis=0)

    return bg.astype(np.float32)


def preprocess(img, bg):
    """
    Convert raw image -> normalized intensity proxy I_norm in [0,1].

    Steps:
    1) background subtraction (clamp negatives to 0 -> absolute black)
    2) median filter to suppress isolated bright tracer particles
    3) robust normalization using percentiles
    4) mild Gaussian smoothing for boundary stability
    """
    x = img.astype(np.float32) - bg
    x[x < 0] = 0

    # Suppress tracer speckles
    if MEDIAN_SIZE and MEDIAN_SIZE > 1:
        x = median_filter(x, size=MEDIAN_SIZE)

    # Robust normalization to [0,1]
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.5)
    x = (x - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0, 1).astype(np.float32)

    # Mild smoothing
    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        x = gaussian(x, sigma=SMOOTH_SIGMA, preserve_range=True).astype(np.float32)

    return x


def intensity_to_concentration(I_norm):
    """Synthetic linear mapping: C [kg/m^3] = I_norm * C_MAX."""
    return (I_norm * C_MAX).astype(np.float32)


# =========================
# Boundary detection: GENERAL contour (outer envelope)
# =========================

def detect_boundary_contour(I_norm):
    """
    Detect the OUTER ENVELOPE as a general contour (not y(x)).

    Pipeline:
    1) threshold at a low onset value
    2) remove small objects + close gaps
    3) keep only the largest connected component (droplet cloud)
    4) extract contours and pick the longest one (outer boundary)

    Returns
    -------
    contour_yx : (N,2) float array of (y, x) points
    mask_main  : 2D bool mask of the selected component
    t          : threshold used
    """
    # Choose onset threshold for "almost vanished" concentration
    if THRESH_METHOD == "otsu":
        # Otsu often too high for outer envelope; scale it down
        t = 0.5 * threshold_otsu(I_norm)
    else:
        t = THRESH_FIXED

    mask = (I_norm > t)
    mask = remove_small_objects(mask, MIN_OBJ_PIXELS)
    mask = binary_closing(mask, disk(CLOSING_RADIUS))

    # Keep largest connected component
    lab = label(mask)
    if lab.max() == 0:
        return None, mask, t

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = np.argmax(counts)
    mask_main = (lab == largest)

    # Extract contour(s) of the blob
    contours = find_contours(mask_main.astype(np.uint8), level=0.5)
    if not contours:
        return None, mask_main, t

    # Longest contour is typically the outer boundary
    contour = max(contours, key=lambda c: c.shape[0]).astype(np.float32)
    return contour, mask_main, t


# =========================
# Diffusive flux along contour (general geometry)
# =========================

def bilinear_sample(img, ys, xs):
    """
    Bilinear sampling of 2D array img at floating-point coordinates (ys, xs).
    ys, xs: arrays of same length.
    """
    H, W = img.shape
    ys = np.clip(ys, 0, H - 1.001)
    xs = np.clip(xs, 0, W - 1.001)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wy = ys - y0
    wx = xs - x0

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return (Ia * (1 - wx) * (1 - wy) +
            Ib * wx * (1 - wy) +
            Ic * (1 - wx) * wy +
            Id * wx * wy)


def diffusive_flux_along_contour(C, contour_yx, D=None):
    """
    Compute diffusive-flux proxy through a GENERAL boundary contour:

        Flux ≈ ∫ ( -D * ∇C · n̂ ) ds

    If D is None, returns the proxy ∫ (-(∇C·n̂)) ds.

    Returns: float
      If D provided: [kg/(m*s)] per unit out-of-plane depth
      If D None: proxy units (use trends)
    """
    # Spatial gradients: dCdy along rows (y), dCdx along cols (x)
    dCdy, dCdx = np.gradient(C)

    ys = contour_yx[:, 0]
    xs = contour_yx[:, 1]

    # Tangent along the contour (in pixel units)
    dy = np.gradient(ys)
    dx = np.gradient(xs)

    seg_len_px = np.sqrt(dx * dx + dy * dy) + 1e-12
    tx = dx / seg_len_px
    ty = dy / seg_len_px

    # Normal = perpendicular to tangent (sign is a convention)
    nx = -ty
    ny = tx

    # Sample gradients at contour points
    gx = bilinear_sample(dCdx, ys, xs)
    gy = bilinear_sample(dCdy, ys, xs)

    # Normal derivative
    dCdn = gx * nx + gy * ny

    # Arc length element in meters
    ds_m = seg_len_px * M_PER_PX

    flux_proxy = float(np.sum((-dCdn) * ds_m))

    if D is None:
        return flux_proxy
    return float(D * flux_proxy)


def mass_in_mask(C, mask_main):
    """
    Mass per unit depth proxy inside droplet cloud:
      M ≈ ∑ C * pixel_area
    where pixel_area = (M_PER_PX)^2 [m^2/px].

    Returns [kg/m] (mass per unit out-of-plane depth), if C is [kg/m^3].
    """
    pixel_area = (M_PER_PX ** 2)
    return float(np.sum(C[mask_main]) * pixel_area)


# =========================
# MAIN
# =========================

idxs, pairs = list_pairs(RAW_DIR)
if len(idxs) == 0:
    raise RuntimeError("No LA/LB tif pairs found. Check RAW_DIR and filenames.")

print(f"Found {len(idxs)} complete LA/LB pairs. Index range: {idxs[0]}..{idxs[-1]}")

bg = estimate_background(idxs, pairs, n_frames=N_BG_FRAMES, method=BG_METHOD)
print("Background estimated.")

# Preload normalized frames (small dataset: 50 pairs -> OK)
frames_I = []
for idx in idxs:
    I = preprocess(read_pair_mean(pairs, idx), bg)
    frames_I.append(I)
frames_I = np.stack(frames_I, axis=0)  # (T, H, W)

T, H, W = frames_I.shape
print(f"Loaded and preprocessed {T} frames. Shape={H}x{W}")

# Update boundary every X frames (dt3)
dt3 = X * dt2_s
times = np.arange(T) * dt2_s

contours = [None] * T
masks = [None] * T
thr_used = [None] * T

current_contour = None
current_mask = None

for i in range(T):
    if (i % X) == 0 or current_contour is None:
        c_new, m_new, thr = detect_boundary_contour(frames_I[i])
        if c_new is not None:
            current_contour = c_new
            current_mask = m_new
        else:
            print(f"Warning: contour detection failed at frame i={i} (idx={idxs[i]}). Keeping previous.")
        thr_used[i] = thr

    contours[i] = current_contour
    masks[i] = current_mask

print(f"Boundary updated every X={X} frames (dt3={dt3} s).")

# Compute time series: mass-in-cloud + diffusive flux proxy
M_series = np.zeros(T, dtype=float)
Jdiff_series = np.zeros(T, dtype=float)

for i in range(T):
    I = frames_I[i]
    C = intensity_to_concentration(I)

    if contours[i] is None or masks[i] is None:
        M_series[i] = np.nan
        Jdiff_series[i] = np.nan
        continue

    M_series[i] = mass_in_mask(C, masks[i])
    Jdiff_series[i] = diffusive_flux_along_contour(C, contours[i], D=D_DIFF)

print("Processing done.")

# =========================
# ANIMATION
# =========================

fig = plt.figure(figsize=(11, 5))
ax_img = fig.add_subplot(1, 2, 1)
ax_plot = fig.add_subplot(1, 2, 2)

im = ax_img.imshow(frames_I[0], cmap="gray", vmin=0, vmax=1)
line_contour, = ax_img.plot([], [], linewidth=1)

ax_img.set_title("Normalized intensity + outer envelope")
ax_img.set_xlim(0, W - 1)
ax_img.set_ylim(H - 1, 0)

ax_plot.set_title("Diffusive flux proxy")
ax_plot.set_xlabel("Time [s]")
ax_plot.set_ylabel("Flux proxy" if D_DIFF is None else "Flux [kg/(m·s)]")

flux_line, = ax_plot.plot([], [], linewidth=1, label="∫ -(∇C·n) ds" if D_DIFF is None else "∫ -D(∇C·n) ds")
flux_dot, = ax_plot.plot([], [], marker="o")

# Optional: also show mass in cloud (secondary line)
mass_line, = ax_plot.plot([], [], linewidth=1, alpha=0.6, label="Mass in cloud [kg/m] (proxy)")
ax_plot.legend(loc="best")

# Axis limits
ax_plot.set_xlim(times.min(), times.max())

# robust y-limits
y = Jdiff_series[np.isfinite(Jdiff_series)]
if y.size == 0:
    ymin, ymax = -1, 1
else:
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if np.isclose(ymin, ymax):
        ymin -= 1
        ymax += 1
pad = 0.1 * (abs(ymax - ymin) + 1e-12)
ax_plot.set_ylim(ymin - pad, ymax + pad)

def update(i):
    im.set_data(frames_I[i])

    c = contours[i]
    if c is not None:
        # contour points are (y, x)
        line_contour.set_data(c[:, 1], c[:, 0])
    else:
        line_contour.set_data([], [])

    ax_img.set_title(f"Idx {idxs[i]} | t={times[i]:.3f} s | thr={thr_used[i] if thr_used[i] is not None else np.nan:.3f}")

    flux_line.set_data(times[:i+1], Jdiff_series[:i+1])
    flux_dot.set_data([times[i]], [Jdiff_series[i]])

    mass_line.set_data(times[:i+1], M_series[:i+1])

    return im, line_contour, flux_line, flux_dot, mass_line

ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)

if SAVE_GIF:
    out_gif = ANALYSIS_DIR / GIF_NAME
    ani.save(out_gif, writer=PillowWriter(fps=FPS))
    print("Saved GIF:", out_gif)

plt.tight_layout()
plt.show()