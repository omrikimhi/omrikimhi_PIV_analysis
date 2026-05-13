#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pointwise turbulent scalar flux and diffusivity on a droplet boundary

Required inputs:
- Paired PLIF images (LA / LB) for scalar concentration estimation.
- PIV vector fields (.VEC files) containing the velocity field.
- Experimental calibration parameters:
    injected volume, injected fluid density, pixel-to-meter ratio, and dt.

Main objective:
Estimate the turbulent scalar transport across the droplet interface by
combining concentration fluctuations with boundary-normal velocity
fluctuations along the detected droplet boundary.

The code computes:
    u_n' c'
and estimates the corresponding turbulent diffusivity:
    D_t,n

on the instantaneous droplet boundary of a selected frame.
----------------------------------------------------------------------

WORKFLOW:

1. Image preprocessing
   - Read paired PLIF images (LA/LB).
   - Average the pair into a single intensity field.
   - Remove background illumination.
   - Apply median and Gaussian filtering.
   - Normalize the intensity field.

2. Concentration estimation
   - Convert image intensity into a concentration proxy.
   - Use the known injected mass and cylindrical-symmetry weighting to scale the concentration field.

3. Boundary detection
   - Detect the droplet region using thresholding.
   - Extract the instantaneous droplet boundary.
   - Smooth and resample the boundary for stable calculations.

4. Velocity processing
   - Read the PIV velocity field from VEC files.
   - Convert the velocity convention to positive downward.
   - Project the velocity onto the local outward boundary normal: U_n

5. Local time averaging
   - Treat each boundary point as a fixed spatial probe.
   - For each point, determine its own local arrival time t0_i:
         first time C_i(t) exceeds a chosen fraction of its local maximum.
   - Compute local time means from t0_i onward: Cbar_i , Unbar_i

6. Turbulent fluctuation calculation
   At the selected frame:
         c'_i  = C_i - Cbar_i
         un'_i = Un_i - Unbar_i

   Compute the local turbulent scalar flux:
         un'_i * c'_i

7. Turbulent diffusivity estimation
   Estimate the local turbulent diffusivity using:
         Dt_n_i = - (un'_i c'_i) / (dCbar/dn)_i

8. Boundary-integrated quantities
   Integrate along the boundary:
         phi_turb = integral(un'c' ds)
         phi_mean = integral(Cbar Unbar ds)

   Compute transport ratios: R and R_total
----------------------------------------------------------------------

MAIN ASSUMPTIONS:

- PLIF intensity is approximately proportional (linearly) to concentration.
- The injected flow rate is approximately steady in time (good assumption).
- Boundary transport is dominated by the boundary-normal component.
- The selected droplet boundary represents the turbulent mixing interface.
- Local time averaging should begin only after scalar arrival at each point.
- The measured image is 2D; conversion to a mass-scaled concentration uses an axisymmetric reconstruction assumption.
- The image contains the full flow field.
----------------------------------------------------------------------

OUTPUTS:

The code produces:

1. Turbulent flux visualization
   - Boundary colored by local u_n' c'
   - Flux variation along boundary arc length

2. Turbulent diffusivity diagnostics
   - Boundary colored by local Dt,n
   - Histogram of Dt,n values
   - Statistical metrics:
         mean
         median
         standard deviation
         percentiles
         IQR

3. Integral transport quantities
   - phi_turb
   - phi_mean
   - R
   - R_total

4. Diagnostic plots and numerical summaries
   for interpretation of turbulent mixing behavior.
"""
# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from skimage.filters import gaussian
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "RawData"
VEC_DIR = BASE_DIR / "Analysis"

TARGET_IDX = 1225 # the frame index to analyze.
USE_MIDDLE_FRAME = False #  If USE_MIDDLE_FRAME=True, this is ignored and the middle frame is used instead.

dt2_s = 1.0

# image preprocessing
N_BG_FRAMES = 5 # Number of background frames to use for background estimation.
BG_METHOD = "median" # Method for background estimation: "median" or "mean".
MEDIAN_SIZE = 3 # Size of the median filter kernel (in pixels). Set to None or 0 to disable.
SMOOTH_SIGMA = 1.0 # Sigma for Gaussian smoothing of the concentration field (in pixels). Set to None or 0 to disable.

# boundary detection
THRESH_METHOD = "fixed" # Method for thresholding: "otsu" for Otsu's method, or "fixed" for a fixed threshold.
THRESH_FIXED = 0.1 # This is a fraction of the normalized intensity (0..1) after preprocessing.
MIN_OBJ_PIXELS = 2000 # Minimum size of detected objects to keep (in pixels). Adjust based on expected droplet size and noise level.
CLOSING_RADIUS = 10 # Radius for binary closing operation (in pixels).
BOUNDARY_RESAMPLE_STEP_PX = 3.0 # Resample the detected boundary points to have approximately this spacing in pixels. Adjust for smoother or more detailed boundaries.

# concentration calibration
INJECTED_VOL_ML = 50.0 # Injected volume in mL
FLUID_DENSITY = 1500.0 # Fluid density in kg/m^3
PX_PER_MM_PLIF = 16.0 # Pixels per mm for PLIF measurement

M_PER_PX_PLIF = 1e-3 / PX_PER_MM_PLIF
PX_AREA_M2 = M_PER_PX_PLIF**2

'''
Choose how the cylindrical symmetry axis x-position is defined:
   "centroid"     = intensity-weighted centroid of the last processed image
   "frame_center" = geometric center of the image frame
   "manual"       = use CYL_MANUAL_AXIS_X_PX
'''
CYL_AXIS_MODE = "centroid"
CYL_MANUAL_AXIS_X_PX = None # used only if CYL_AXIS_MODE = "manual", e.g. 512.0
CYL_R_MIN_PX = 0.5 # avoids zero volume weight exactly on the symmetry axis

# local time average after arrival
ARRIVAL_FRAC = 0.05 # Fraction of local max to define arrival time (t_0)
MIN_MEAN_SAMPLES = 3 # Minimum number of samples to compute mean

# gradient for Dt_n
NORMAL_OFFSET_PX = 6.0 # Offset for normal vector calculation (in pixels) [dC/dn]
GRAD_EPS = 1e-12 # Epsilon for gradient calculation
R_EPS = 1e-20 # Epsilon for radius calculation

INVALID_CHC_VALUES = {0} # Values that are considered invalid for concentration from the VEC files.

#cache
VEC_CACHE = {} # Cache for preloaded VEC files. This avoids reading and pivoting each .vec file again inside the main time loop.


# contour smoothing
SMOOTH_CONTOUR_SIGMA_PX = 10.0 # Larger values make the detected boundary smoother and less wiggly.

# along-boundary signal smoothing / outlier reduction
# These are used for visualization and the final reported integrals.
APPLY_SIGNAL_SMOOTHING = True # Set False if you want the raw unsmoothed values.
SIGNAL_SMOOTH_SIGMA_POINTS = 2.0 # Sigma for Gaussian smoothing of along-boundary signals (in number of points). Set to None or 0 to disable.
CLIP_SIGNAL_PERCENTILES = (1, 99) # Percentiles for cutting along-boundary signals before smoothing.

# plotting controls
SHOW_NORMAL_ARROWS = True
NORMAL_ARROW_COUNT = 40 # Number of normal arrows to show on the boundary plot. Adjust for clarity.

# Better color scaling for Dt,n map:
# percentile range avoids a few extreme outliers crushing the colorbar.
DT_COLOR_PERCENTILES = (2, 98) # Percentiles for Dt,n color scaling.
FLUX_COLOR_PERCENTILE = 98 # Percentile for u_n'c' color scaling.
HIST_BINS = 60 # Number of bins for Dt,n histogram.
HIST_LOG_Y = True # Whether to use log scale for Dt,n histogram.


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def list_pairs(raw_dir):
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    pairs = {}

    for f in sorted(files):
        name_u = f.name.upper()
        if ".LA." in name_u:
            ab = "LA"
        elif ".LB." in name_u:
            ab = "LB"
        else:
            continue

        m = re.search(r"_(\d{6})\.", f.name)
        if not m:
            continue

        idx = int(m.group(1))
        pairs.setdefault(idx, {})[ab] = f

    idxs = sorted([k for k, v in pairs.items() if "LA" in v and "LB" in v])
    return idxs, pairs


def read_pair_mean(pairs, idx):
    A = io.imread(pairs[idx]["LA"]).astype(np.float32)
    B = io.imread(pairs[idx]["LB"]).astype(np.float32)
    return 0.5 * (A + B)


def estimate_background(idxs, pairs):
    n = min(N_BG_FRAMES, len(idxs))
    stack = np.stack([read_pair_mean(pairs, idx) for idx in idxs[:n]], axis=0)

    if BG_METHOD == "median":
        return np.median(stack, axis=0).astype(np.float32)
    return np.mean(stack, axis=0).astype(np.float32)


def preprocess(img, bg):
    x = img.astype(np.float32) - bg
    x[x < 0] = 0

    if MEDIAN_SIZE and MEDIAN_SIZE > 1:
        x = median_filter(x, size=MEDIAN_SIZE)

    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.5)
    x = (x - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0, 1).astype(np.float32)

    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        x = gaussian(x, sigma=SMOOTH_SIGMA, preserve_range=True).astype(np.float32)

    return x


def determine_cyl_axis_x(I_last):
    """Determine the x-position of the cylindrical symmetry axis.

    Modes:
    - centroid: intensity-weighted centroid of the last processed image.
    - frame_center: geometric center of the image frame.
    - manual: user-defined CYL_MANUAL_AXIS_X_PX.
    """
    H, W = I_last.shape

    if CYL_AXIS_MODE == "frame_center":
        return 0.5 * (W - 1)

    if CYL_AXIS_MODE == "manual":
        if CYL_MANUAL_AXIS_X_PX is None:
            raise RuntimeError("CYL_AXIS_MODE='manual' requires CYL_MANUAL_AXIS_X_PX to be set.")
        return float(CYL_MANUAL_AXIS_X_PX)

    if CYL_AXIS_MODE == "centroid":
        x = np.arange(W, dtype=float)
        weights_x = np.nansum(I_last, axis=0)
        total = float(np.nansum(weights_x))
        if total < 1e-12:
            raise RuntimeError("Cannot determine cylindrical axis: last-frame intensity is too small.")
        return float(np.nansum(x * weights_x) / total)

    raise RuntimeError("Invalid CYL_AXIS_MODE. Use 'centroid', 'frame_center', or 'manual'.")


def cylindrical_volume_weights(shape, x_axis_px):
    """Return per-pixel volume weights for a full axisymmetric PLIF slice.

    The image is interpreted as a vertical x-y slice through an approximately
    axisymmetric droplet/plume. The x distance from the symmetry axis is the
    radius r.

    This version always assumes that both sides of the flow are visible in the
    image. Therefore each pixel receives pi*r*dA, because the left and right
    sides together represent the full annulus 2*pi*r*dr*dy.
    """
    H, W = shape
    x = np.arange(W, dtype=float)
    r_px = np.maximum(np.abs(x - x_axis_px), CYL_R_MIN_PX)
    r_m = r_px * M_PER_PX_PLIF

    weights_1d = np.pi * r_m * PX_AREA_M2
    return np.broadcast_to(weights_1d[None, :], (H, W)).astype(np.float64)


def calibrate_k(idxs, pairs, bg, x_axis_px):
    """Scale intensity to concentration using cylindrical symmetry.

    C = k*I, and k is chosen so that integral(C dV) equals the injected mass.
    """
    V_m3 = INJECTED_VOL_ML * 1e-6
    M_total_kg = FLUID_DENSITY * V_m3

    I_last = preprocess(read_pair_mean(pairs, idxs[-1]), bg)
    dV = cylindrical_volume_weights(I_last.shape, x_axis_px)

    weighted_intensity = float(np.nansum(I_last * dV))
    if weighted_intensity < 1e-12:
        raise RuntimeError("Last frame weighted intensity is too small for calibration.")

    return M_total_kg / weighted_intensity


def sample_image(img, x_px, y_px):
    H, W = img.shape
    interp = RegularGridInterpolator(
        (np.arange(H, dtype=float), np.arange(W, dtype=float)),
        img,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([y_px, x_px]))


def sample_image_triplet(img, x0, y0, x1, y1, x2, y2):
    """Sample one image at three point sets using one interpolator.

    This is faster than building a new RegularGridInterpolator three times
    for C, Cout, and Cin.
    """
    H, W = img.shape
    interp = RegularGridInterpolator(
        (np.arange(H, dtype=float), np.arange(W, dtype=float)),
        img,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pts0 = np.column_stack([y0, x0])
    pts1 = np.column_stack([y1, x1])
    pts2 = np.column_stack([y2, x2])

    return interp(pts0), interp(pts1), interp(pts2)


# ============================================================
# BOUNDARY FUNCTIONS
# ============================================================

def detect_boundary(I):
    if THRESH_METHOD == "otsu":
        from skimage.filters import threshold_otsu
        thr = 0.5 * threshold_otsu(I)
    else:
        thr = THRESH_FIXED

    mask = I > thr
    mask = remove_small_objects(mask, MIN_OBJ_PIXELS)
    mask = binary_closing(mask, disk(CLOSING_RADIUS))

    lab = label(mask)
    if lab.max() == 0:
        return None, mask, thr

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    main = lab == np.argmax(counts)

    contours = find_contours(main.astype(np.uint8), level=0.5)
    if not contours:
        return None, main, thr

    return max(contours, key=lambda c: c.shape[0]).astype(np.float32), main, thr


def smooth_closed_contour(contour_yx, sigma_px=2.0):
    """Smooth a closed contour in pixel coordinates.

    The contour points are treated as periodic, so the start/end of the
    contour remain continuous. This reduces small wiggles from threshold noise.
    """
    if sigma_px is None or sigma_px <= 0:
        return contour_yx.copy()

    y = contour_yx[:, 0]
    x = contour_yx[:, 1]

    y_s = gaussian_filter1d(y, sigma=sigma_px, mode="wrap")
    x_s = gaussian_filter1d(x, sigma=sigma_px, mode="wrap")

    return np.column_stack([y_s, x_s]).astype(np.float32)


def resample_contour(contour_yx, step_px):
    y = contour_yx[:, 0]
    x = contour_yx[:, 1]

    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate([[0], np.cumsum(ds)])

    if s[-1] < step_px:
        return contour_yx.copy()

    s_new = np.arange(0, s[-1], step_px)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)

    return np.column_stack([y_new, x_new]).astype(np.float32)


def contour_geometry(contour_yx):
    y = contour_yx[:, 0]
    x = contour_yx[:, 1]

    dy = np.gradient(y)
    dx = np.gradient(x)
    ds_px = np.sqrt(dx**2 + dy**2) + 1e-12

    tx = dx / ds_px
    ty = dy / ds_px

    nx = -ty
    ny = tx

    # orient normal approximately outward using centroid
    rx = x - np.mean(x)
    ry = y - np.mean(y)
    sign = np.sign(nx * rx + ny * ry)
    sign[sign == 0] = 1
    nx *= sign
    ny *= sign

    return x, y, nx, ny, ds_px


# ============================================================
# VEC FUNCTIONS
# ============================================================

def resolve_vec_path(idx):
    files = sorted(VEC_DIR.glob(f"*_{idx:06d}*.vec"))
    if not files:
        return None
    return files[0]


def parse_vec_header(path):
    header = path.read_text(errors="ignore").splitlines()[0]

    def grab(pattern, default=np.nan):
        m = re.search(pattern, header)
        return float(m.group(1)) if m else default

    return {
        "um_per_px_x": grab(r'MicrometersPerPixelX="([0-9.]+)"'),
        "um_per_px_y": grab(r'MicrometersPerPixelY="([0-9.]+)"'),
        "origin_x_px": grab(r'OriginInImageX="([0-9.]+)"'),
        "origin_y_px": grab(r'OriginInImageY="([0-9.]+)"'),
    }


def read_vec_grid(path):
    df = pd.read_csv(path, sep=r"[,\s]+", skiprows=1, header=None, engine="python")
    df.columns = ["x", "y", "u", "v", "chc", "unc_low", "unc_high"]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.loc[df["chc"].isin(INVALID_CHC_VALUES), ["u", "v"]] = np.nan

    xu = np.sort(df["x"].dropna().unique())
    yu = np.sort(df["y"].dropna().unique())

    U = df.pivot(index="y", columns="x", values="u").sort_index().to_numpy(dtype=float)
    V = df.pivot(index="y", columns="x", values="v").sort_index().to_numpy(dtype=float)

    return xu, yu, U, V


def preload_vec_data(idxs):
    """Load all VEC files once into memory.

    This avoids repeating the slow operations inside the main loop:
    disk reading, pandas parsing, pivoting, and interpolator construction.
    """
    global VEC_CACHE
    VEC_CACHE = {}

    for idx in idxs:
        path = resolve_vec_path(idx)
        if path is None:
            continue

        meta = parse_vec_header(path)
        xu, yu, U, V = read_vec_grid(path)

        ui = RegularGridInterpolator((yu, xu), U, bounds_error=False, fill_value=np.nan)
        vi = RegularGridInterpolator((yu, xu), V, bounds_error=False, fill_value=np.nan)

        VEC_CACHE[idx] = {
            "meta": meta,
            "ui": ui,
            "vi": vi,
            "mm_per_px_x": meta["um_per_px_x"] * 1e-3,
            "mm_per_px_y": meta["um_per_px_y"] * 1e-3,
        }

    print(f"Loaded {len(VEC_CACHE)} VEC fields into memory.")


def sample_unormal(idx, x_px, y_px, nx, ny):
    if idx not in VEC_CACHE:
        return np.full_like(x_px, np.nan, dtype=float)

    data = VEC_CACHE[idx]
    meta = data["meta"]
    ui = data["ui"]
    vi = data["vi"]

    x_mm = (x_px - meta["origin_x_px"]) * data["mm_per_px_x"]
    y_mm = -(y_px - meta["origin_y_px"]) * data["mm_per_px_y"]

    pts = np.column_stack([y_mm, x_mm])

    u = ui(pts)
    v_down = -vi(pts)  # VEC v to image-y-positive-down convention

    return u * nx + v_down * ny


# ============================================================
# STATISTICS
# ============================================================

def mean_after_local_arrival(C_ts):
    T, N = C_ts.shape
    Cbar = np.full(N, np.nan)
    t0 = np.full(N, -1, dtype=int)
    ok = np.zeros(N, dtype=bool)

    for i in range(N):
        s = C_ts[:, i]

        if np.count_nonzero(np.isfinite(s)) < MIN_MEAN_SAMPLES:
            continue

        mx = np.nanmax(s)
        if not np.isfinite(mx) or mx <= 0:
            continue

        ids = np.where(np.isfinite(s) & (s >= ARRIVAL_FRAC * mx))[0]
        if ids.size == 0:
            continue

        i0 = int(ids[0])
        tail = s[i0:]
        tail = tail[np.isfinite(tail)]

        if tail.size < MIN_MEAN_SAMPLES:
            continue

        Cbar[i] = np.mean(tail)
        t0[i] = i0
        ok[i] = True

    return Cbar, t0, ok


def tail_mean_from_t0(X_ts, t0):
    N = X_ts.shape[1]
    out = np.full(N, np.nan)

    for i in range(N):
        if t0[i] < 0:
            continue

        tail = X_ts[t0[i]:, i]
        tail = tail[np.isfinite(tail)]

        if tail.size >= MIN_MEAN_SAMPLES:
            out[i] = np.mean(tail)

    return out


def safe_percentile(values, percentiles, fallback=(-1.0, 1.0)):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    out = np.nanpercentile(values, percentiles)
    if np.any(~np.isfinite(out)) or np.isclose(out[0], out[1]):
        return fallback
    return out


def clip_and_smooth_periodic(values, valid, clip_percentiles=(1, 99), sigma_points=2.0):
    """
    Clip outliers and smooth a signal along a closed boundary.

    This is meant for along-boundary quantities such as u_n'c' or D_t,n.
    It preserves NaNs outside valid points and smooths only after clipping
    strong outliers. The boundary is treated as periodic.
    """
    y = np.asarray(values, dtype=float).copy()
    out = np.full_like(y, np.nan, dtype=float)

    vv = valid & np.isfinite(y)
    if np.count_nonzero(vv) < 5:
        return y

    lo, hi = np.nanpercentile(y[vv], clip_percentiles)
    y_clip = y.copy()
    y_clip[vv] = np.clip(y_clip[vv], lo, hi)

    idx = np.arange(len(y_clip))
    y_filled = y_clip.copy()
    if np.any(~vv):
        y_filled[~vv] = np.interp(idx[~vv], idx[vv], y_clip[vv])

    if sigma_points and sigma_points > 0:
        y_smooth = gaussian_filter1d(y_filled, sigma=sigma_points, mode="wrap")
    else:
        y_smooth = y_filled

    out[vv] = y_smooth[vv]
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    idxs, pairs = list_pairs(RAW_DIR)
    if not idxs:
        raise RuntimeError("No LA/LB pairs found.")

    target_idx = idxs[len(idxs)//2] if USE_MIDDLE_FRAME else TARGET_IDX
    if target_idx not in pairs:
        raise RuntimeError(f"TARGET_IDX={target_idx} not found.")

    print(f"Found {len(idxs)} frames: {idxs[0]}..{idxs[-1]}")
    print(f"Selected frame: {target_idx}")

    print("Loading VEC files into memory...")
    preload_vec_data(idxs)

    bg = estimate_background(idxs, pairs)

    # First detect the boundary on the selected frame. All later statistics are
    # computed only at these boundary points and at their normal offsets.
    I0 = preprocess(read_pair_mean(pairs, target_idx), bg)
    contour_raw, mask, thr = detect_boundary(I0)
    if contour_raw is None:
        raise RuntimeError("Boundary detection failed.")

    contour_raw_smooth = smooth_closed_contour(contour_raw, sigma_px=SMOOTH_CONTOUR_SIGMA_PX)
    contour = resample_contour(contour_raw_smooth, BOUNDARY_RESAMPLE_STEP_PX)
    x, y, nx, ny, ds_px = contour_geometry(contour)
    Np = len(x)

    I_last_for_axis = preprocess(read_pair_mean(pairs, idxs[-1]), bg)
    x_axis_px = determine_cyl_axis_x(I_last_for_axis)
    k = calibrate_k(idxs, pairs, bg, x_axis_px)
    print(f"Cylindrical axis mode = {CYL_AXIS_MODE}")
    print(f"Cylindrical symmetry axis x = {x_axis_px:.2f} px")
    print(f"Calibration k = {k:.6e} kg/m^3 per normalized intensity")

    if target_idx not in VEC_CACHE:
        raise RuntimeError("No VEC file for selected frame.")

    meta0 = VEC_CACHE[target_idx]["meta"]
    mm_per_px = 0.5 * (meta0["um_per_px_x"] + meta0["um_per_px_y"]) * 1e-3
    ds_m = ds_px * mm_per_px * 1e-3
    dn_m = NORMAL_OFFSET_PX * mm_per_px * 1e-3

    x_out = x + NORMAL_OFFSET_PX * nx
    y_out = y + NORMAL_OFFSET_PX * ny
    x_in = x - NORMAL_OFFSET_PX * nx
    y_in = y - NORMAL_OFFSET_PX * ny

    Nt = len(idxs)
    C_ts = np.full((Nt, Np), np.nan)
    Un_ts = np.full((Nt, Np), np.nan)
    Cout_ts = np.full((Nt, Np), np.nan)
    Cin_ts = np.full((Nt, Np), np.nan)

    for it, idx in enumerate(idxs):
        I = preprocess(read_pair_mean(pairs, idx), bg)
        C = k * I

        C_ts[it], Cout_ts[it], Cin_ts[it] = sample_image_triplet(
            C, x, y, x_out, y_out, x_in, y_in
        )
        Un_ts[it] = sample_unormal(idx, x, y, nx, ny)

        if (it + 1) % 10 == 0 or it == Nt - 1:
            print(f"Processed {it+1}/{Nt}")

    target_i = idxs.index(target_idx)

    Cbar, t0_local, ok_c = mean_after_local_arrival(C_ts)
    Unbar = tail_mean_from_t0(Un_ts, t0_local)
    Cout_bar = tail_mean_from_t0(Cout_ts, t0_local)
    Cin_bar = tail_mean_from_t0(Cin_ts, t0_local)

    C_now = C_ts[target_i]
    Un_now = Un_ts[target_i]

    c_prime = C_now - Cbar
    un_prime = Un_now - Unbar
    uc_turb = un_prime * c_prime

    dCdn = (Cout_bar - Cin_bar) / (2 * dn_m + GRAD_EPS)
    Dt_n = -uc_turb / (dCdn + np.sign(dCdn) * GRAD_EPS)

    valid = (
        ok_c &
        (target_i >= t0_local) &
        np.isfinite(C_now) &
        np.isfinite(Un_now) &
        np.isfinite(Cbar) &
        np.isfinite(Unbar) &
        np.isfinite(uc_turb) &
        np.isfinite(Dt_n) &
        np.isfinite(ds_m)
    )

    if np.count_nonzero(valid) < 10:
        raise RuntimeError("Too few valid boundary points. Try a later TARGET_IDX or check VEC overlap.")

    # Optional along-boundary smoothing and outlier clipping.
    # These processed arrays are used for the displayed maps and final scalar outputs.
    if APPLY_SIGNAL_SMOOTHING:
        uc_turb_plot = clip_and_smooth_periodic(
            uc_turb,
            valid,
            clip_percentiles=CLIP_SIGNAL_PERCENTILES,
            sigma_points=SIGNAL_SMOOTH_SIGMA_POINTS,
        )
        Dt_n_plot = clip_and_smooth_periodic(
            Dt_n,
            valid,
            clip_percentiles=CLIP_SIGNAL_PERCENTILES,
            sigma_points=SIGNAL_SMOOTH_SIGMA_POINTS,
        )
    else:
        uc_turb_plot = uc_turb.copy()
        Dt_n_plot = Dt_n.copy()

    phi_turb = np.nansum(uc_turb_plot[valid] * ds_m[valid])
    phi_mean = np.nansum(Cbar[valid] * Unbar[valid] * ds_m[valid])
    R = abs(phi_turb) / (abs(phi_mean) + R_EPS)

    total_abs = np.nansum(np.abs(C_now[valid] * Un_now[valid]) * ds_m[valid])
    R_total = abs(phi_turb) / (total_abs + R_EPS)

    D_valid = Dt_n_plot[valid]
    D_valid = D_valid[np.isfinite(D_valid)]

    D_mean = np.nanmean(D_valid)
    D_median = np.nanmedian(D_valid)
    D_std = np.nanstd(D_valid)

    print("\n===== RESULTS =====")
    print(f"Valid boundary points: {np.count_nonzero(valid)*100/Np:.1f}%")
    print(f"phi_turb = {phi_turb:.6e}")
    print(f"phi_mean = {phi_mean:.6e}")
    print(f"R = {R:.6e}")
    print(f"R_total = {R_total:.6e}")
    print(f"Dt_n mean = {D_mean:.6e} m^2/s")
    print(f"Dt_n median = {D_median:.6e} m^2/s")
    print(f"Dt_n std = {D_std:.6e} m^2/s")
    print(f"Signal smoothing: {APPLY_SIGNAL_SMOOTHING}, sigma={SIGNAL_SMOOTH_SIGMA_POINTS}, clip={CLIP_SIGNAL_PERCENTILES}")

    # ========================================================
    # PLOTS - split by physical quantity
    # ========================================================

    s = np.concatenate([[0], np.cumsum(ds_m[:-1])])

    # --------------------------------------------------------
    # Figure 1: Turbulent flux only
    # --------------------------------------------------------
    fig_flux = plt.figure(figsize=(15, 7))
    gs_flux = fig_flux.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.38)

    ax1 = fig_flux.add_subplot(gs_flux[0, 0])
    ax1.imshow(I0, cmap="gray", vmin=0, vmax=1)
    ax1.plot(contour[:, 1], contour[:, 0], color="white", lw=0.8, alpha=0.7)

    flux_lim = np.nanpercentile(np.abs(uc_turb_plot[valid]), FLUX_COLOR_PERCENTILE)
    if not np.isfinite(flux_lim) or flux_lim == 0:
        flux_lim = 1.0

    sc1 = ax1.scatter(
        x[valid], y[valid],
        c=uc_turb_plot[valid],
        s=18,
        cmap="coolwarm",
        vmin=-flux_lim,
        vmax=flux_lim,
        edgecolors="none",
    )

    if SHOW_NORMAL_ARROWS:
        step = max(1, len(x) // NORMAL_ARROW_COUNT)
        ax1.quiver(
            x[::step], y[::step],
            nx[::step], ny[::step],
            color="lime",
            scale=20,
            width=0.004,
        )

    ax1.set_title("Boundary colored by local turbulent flux: $u_n'c'$")
    ax1.set_xlabel("x [px]")
    ax1.set_ylabel("y [px], positive downward")
    ax1.set_xlim(0, I0.shape[1] - 1)
    ax1.set_ylim(I0.shape[0] - 1, 0)
    cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("$u_n'c'$")

    ax2 = fig_flux.add_subplot(gs_flux[0, 1])
    ax2.plot(s[valid], uc_turb_plot[valid], lw=1.2, label="$u_n'c'$")
    ax2.axhline(0, color="gray", ls="--")
    ax2.set_xlabel("arc length [m]")
    ax2.set_ylabel("local turbulent flux")
    ax2.set_title("Pointwise turbulent flux along boundary")
    ax2.legend()

    fig_flux.suptitle(
        f"Turbulent scalar flux on selected boundary | "
        f"phi_turb={phi_turb:.3e}, phi_mean={phi_mean:.3e}, R={R:.3g}",
        fontsize=15,
    )
    fig_flux.tight_layout()


    # --------------------------------------------------------
    # Figure 2: Turbulent diffusivity only
    # --------------------------------------------------------
    fig_diff = plt.figure(figsize=(16, 8))
    gs_diff = fig_diff.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0],
                                    wspace=0.30, hspace=0.34)

    # Dt,n spatial map
    ax3 = fig_diff.add_subplot(gs_diff[0, 0])
    ax3.imshow(I0, cmap="gray", vmin=0, vmax=1)
    ax3.plot(contour[:, 1], contour[:, 0], color="white", lw=0.8, alpha=0.7)

    dt_vmin, dt_vmax = safe_percentile(D_valid, DT_COLOR_PERCENTILES)
    if dt_vmin < 0 < dt_vmax:
        dt_abs = max(abs(dt_vmin), abs(dt_vmax))
        cmap_dt = "coolwarm"
        dt_vmin, dt_vmax = -dt_abs, dt_abs
    else:
        cmap_dt = "viridis"

    sc3 = ax3.scatter(
        x[valid], y[valid],
        c=Dt_n_plot[valid],
        s=18,
        cmap=cmap_dt,
        vmin=dt_vmin,
        vmax=dt_vmax,
        edgecolors="none",
    )

    ax3.set_title("Estimated local turbulent diffusivity $D_{t,n}$")
    ax3.set_xlabel("x [px]")
    ax3.set_ylabel("y [px], positive downward")
    ax3.set_xlim(0, I0.shape[1] - 1)
    ax3.set_ylim(I0.shape[0] - 1, 0)
    cb3 = plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label("$D_{t,n}$ [m²/s]")

    # Dt,n histogram
    ax4 = fig_diff.add_subplot(gs_diff[1, 0])
    hist_range = safe_percentile(D_valid, DT_COLOR_PERCENTILES)
    ax4.hist(D_valid, bins=HIST_BINS, range=hist_range, log=HIST_LOG_Y)
    ax4.axvline(D_mean, color="black", ls="--", lw=1.2, label="mean")
    ax4.axvline(D_median, color="gray", ls=":", lw=1.8, label="median")
    ax4.set_xlabel("$D_{t,n}$ [m²/s]")
    ax4.set_ylabel("count" + (" (log scale)" if HIST_LOG_Y else ""))
    ax4.set_title("Histogram of local turbulent diffusivity")
    ax4.legend()

    # Diffusivity summary box
    ax5 = fig_diff.add_subplot(gs_diff[:, 1])
    ax5.axis("off")

    t0v = t0_local[valid]
    diff_summary = (
        f"Selected idx: {target_idx}\n"
        f"Valid points: {np.count_nonzero(valid)} / {Np} "
        f"({100*np.count_nonzero(valid)/Np:.1f}%)\n"
        f"Local t0 range: {np.min(t0v):.0f}..{np.max(t0v):.0f} frames\n"
        f"Local t0 median: {np.median(t0v):.0f} frames\n\n"
        f"Turbulent diffusivity statistics\n"
        f"-----------------------------\n"
        f"Dt,n mean   = {D_mean:.4e} m²/s\n"
        f"Dt,n median = {D_median:.4e} m²/s\n"
        f"Dt,n std    = {D_std:.4e} m²/s\n"
        f"Related flux integrals\n"
        f"-----------------------------\n"
        f"phi_turb = {phi_turb:.4e}\n"
        f"phi_mean = {phi_mean:.4e}\n"
        f"R        = {R:.4e}\n"
        f"R_total  = {R_total:.4e}\n\n"
        f"Plot note\n"
        f"-----------------------------\n"
        f"Dt,n map and histogram use percentile scaling: {DT_COLOR_PERCENTILES}.\n"
        f"Extreme outliers may lie outside the displayed color limits.\n\n"
        f"Smoothing note\n"
        f"-----------------------------\n"
        f"Boundary smoothing sigma = {SMOOTH_CONTOUR_SIGMA_PX}\n"
        f"Signal smoothing = {APPLY_SIGNAL_SMOOTHING}\n"
        f"Signal sigma = {SIGNAL_SMOOTH_SIGMA_POINTS}\n"
        f"Signal clipping = {CLIP_SIGNAL_PERCENTILES}\n\n"
        f"Convention\n"
        f"-----------------------------\n"
        f"image y and plotted velocity are positive downward.\n"
        f"Un is positive along the selected-frame outward normal."
    )

    ax5.text(
        0.02, 0.98,
        diff_summary,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.25,
        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"),
    )

    fig_diff.suptitle("Turbulent diffusivity diagnostics on selected boundary", fontsize=15)

    # Show both windows together.
    plt.show()


if __name__ == "__main__":
    main()