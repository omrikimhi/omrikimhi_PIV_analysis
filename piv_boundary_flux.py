#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-series turbulent scalar flux across the droplet interface
-------------------------------------------------------------

What this script does
---------------------
For ALL available PLIF/PIV frames:
1) Reads each LA/LB PLIF pair and preprocesses it like in 2_point_correlation.py
2) Detects the instantaneous droplet boundary like in data_analist_PIV.py
3) Reads the matching PIV .vec field
4) Samples concentration C and normal velocity W_n along the instantaneous boundary
5) Computes the line-integrated transport at each frame:
       J_cw(t) = ∮ C(s,t) * W_n(s,t) ds
6) Computes also the ds-weighted means along the boundary for each frame:
       C_bar_b(t), W_bar_b(t)
7) Over a chosen quasi-steady time window [t0, end], estimates the turbulent part via
       <cw>_t - <C>_t <W>_t
   and in line-integrated form:
       J_turb = <J_cw>_t - <L*Cbar*Wbar>_t
   where L is the valid boundary length in each frame.

Important physical note
-----------------------
This gives a PRACTICAL estimate of turbulent scalar transport through the
instantaneous droplet interface without trying to track the same boundary point
across time. It is therefore suitable for your current experiment where droplets
enter at a statistically steady rate.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from skimage.filters import gaussian
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import median_filter
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "RawData"
VEC_DIR = BASE_DIR / "Analysis"

# ----- Timing -----
dt2_s = 1.0

# ----- Image preprocessing -----
N_BG_FRAMES = 5
BG_METHOD = "median"
MEDIAN_SIZE = 3
SMOOTH_SIGMA = 1.0

# ----- Boundary detection -----
THRESH_METHOD = "fixed"
THRESH_FIXED = 0.05
MIN_OBJ_PIXELS = 2000
CLOSING_RADIUS = 2
BOUNDARY_RESAMPLE_STEP_PX = 3.0

# ----- Concentration calibration -----
INJECTED_VOL_ML = 50.0
FLUID_DENSITY = 1000.0      # kg/m^3
DEPTH_M = 0.1
PX_PER_MM_PLIF = 16.0
M_PER_PX_PLIF = 1e-3 / PX_PER_MM_PLIF
PX_AREA_M2 = M_PER_PX_PLIF**2

# ----- VEC parsing -----
INVALID_CHC_VALUES = {0}

# ----- Quasi-steady time-window selection -----
# The time average starts at the first frame where the total image mass proxy
# reaches MASS_THRESHOLD_FRAC of its maximum.
MASS_THRESHOLD_FRAC = 0.05
MANUAL_T0_IDX = None   # set integer to override auto detection

# ----- Plotting -----
QUIVER_SKIP = 2
SHOW_EXAMPLE_FRAME = True
EXAMPLE_FRAME_MODE = 1225  # "t0", "middle", or integer frame index


# ============================================================
# FILE HELPERS
# ============================================================

def list_pairs(raw_dir):
    files = [p for p in raw_dir.iterdir()
             if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]

    pairs = {}
    for f in sorted(files):
        nameU = f.name.upper()
        if ".LA." in nameU:
            ab = "LA"
        elif ".LB." in nameU:
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


def estimate_background(idxs, pairs, n_frames=5, method="median"):
    n = min(n_frames, len(idxs))
    stack = [read_pair_mean(pairs, idx) for idx in idxs[:n]]
    stack = np.stack(stack, axis=0)
    if method == "median":
        bg = np.median(stack, axis=0)
    else:
        bg = np.mean(stack, axis=0)
    return bg.astype(np.float32)


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


# ============================================================
# BOUNDARY DETECTION
# ============================================================

def detect_boundary_contour(I_norm):
    if THRESH_METHOD == "otsu":
        from skimage.filters import threshold_otsu
        t = 0.5 * threshold_otsu(I_norm)
    else:
        t = THRESH_FIXED

    mask = (I_norm > t)
    mask = remove_small_objects(mask, MIN_OBJ_PIXELS)
    mask = binary_closing(mask, disk(CLOSING_RADIUS))

    lab = label(mask)
    if lab.max() == 0:
        return None, mask, t

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = np.argmax(counts)
    mask_main = (lab == largest)

    contours = find_contours(mask_main.astype(np.uint8), level=0.5)
    if not contours:
        return None, mask_main, t

    contour_yx = max(contours, key=lambda c: c.shape[0]).astype(np.float32)
    return contour_yx, mask_main, t


def resample_contour_by_arclength(contour_yx, step_px=3.0):
    ys = contour_yx[:, 0]
    xs = contour_yx[:, 1]
    ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    if s[-1] < step_px:
        return contour_yx.copy()

    s_new = np.arange(0.0, s[-1], step_px)
    x_new = np.interp(s_new, s, xs)
    y_new = np.interp(s_new, s, ys)
    return np.column_stack([y_new, x_new]).astype(np.float32)


def contour_geometry(contour_yx):
    ys = contour_yx[:, 0]
    xs = contour_yx[:, 1]

    dy = np.gradient(ys)
    dx = np.gradient(xs)
    seg_len_px = np.sqrt(dx**2 + dy**2) + 1e-12

    tx = dx / seg_len_px
    ty = dy / seg_len_px
    nx = -ty
    ny = tx

    x_c = np.mean(xs)
    y_c = np.mean(ys)
    rx = xs - x_c
    ry = ys - y_c
    sign = np.sign(nx * rx + ny * ry)
    sign[sign == 0] = 1.0
    nx *= sign
    ny *= sign

    return {
        "x": xs,
        "y": ys,
        "tx": tx,
        "ty": ty,
        "nx": nx,
        "ny": ny,
        "ds_px": seg_len_px,
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_k_from_last_frame(idxs, pairs, bg):
    V_m3 = INJECTED_VOL_ML * 1e-6
    M_total_kg = FLUID_DENSITY * V_m3

    I_last = preprocess(read_pair_mean(pairs, idxs[-1]), bg)
    sum_I_last = float(np.sum(I_last))
    if sum_I_last < 1e-12:
        raise RuntimeError("Last frame intensity sum is too small for calibration.")

    k = M_total_kg / (sum_I_last * PX_AREA_M2 * DEPTH_M)
    return k


# ============================================================
# VEC READING
# ============================================================

def read_insight_vec(file_path):
    df = pd.read_csv(
        file_path,
        sep=r"[,\s]+",
        skiprows=1,
        header=None,
        engine="python"
    )
    df.columns = ["x", "y", "u", "v", "chc", "unc_low", "unc_high"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_vec_header(file_path):
    header = file_path.read_text(errors="ignore").splitlines()[0]

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, header)
        if m:
            return cast(m.group(1))
        return default

    meta = {
        "img_width_px": grab(r'SourceImageWidth="([0-9.]+)"'),
        "img_height_px": grab(r'SourceImageHeight="([0-9.]+)"'),
        "um_per_px_x": grab(r'MicrometersPerPixelX="([0-9.]+)"'),
        "um_per_px_y": grab(r'MicrometersPerPixelY="([0-9.]+)"'),
        "origin_x_px": grab(r'OriginInImageX="([0-9.]+)"'),
        "origin_y_px": grab(r'OriginInImageY="([0-9.]+)"'),
        "dt_us": grab(r'MicrosecondsPerDeltaT="([0-9.]+)"'),
    }
    return meta


def build_vec_grids(vec_path):
    df = read_insight_vec(vec_path)
    df.loc[df["chc"].isin(INVALID_CHC_VALUES), ["u", "v"]] = np.nan

    x_unique = np.sort(df["x"].dropna().unique())
    y_unique = np.sort(df["y"].dropna().unique())

    U = df.pivot(index="y", columns="x", values="u").sort_index().to_numpy(dtype=float)
    V = df.pivot(index="y", columns="x", values="v").sort_index().to_numpy(dtype=float)

    return x_unique, y_unique, U, V


def bilinear_interpolator(x_unique, y_unique, field):
    return RegularGridInterpolator(
        (y_unique, x_unique),
        field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


# ============================================================
# MAPPING AND FRAME CALCULATION
# ============================================================

def contour_pixels_to_vec_mm(contour_geom, vec_meta):
    mm_per_px_x = vec_meta["um_per_px_x"] * 1e-3
    mm_per_px_y = vec_meta["um_per_px_y"] * 1e-3

    x_mm = (contour_geom["x"] - vec_meta["origin_x_px"]) * mm_per_px_x
    y_mm = - (contour_geom["y"] - vec_meta["origin_y_px"]) * mm_per_px_y
    ds_m = contour_geom["ds_px"] * (0.5 * (mm_per_px_x + mm_per_px_y)) * 1e-3
    return x_mm, y_mm, ds_m


def resolve_vec_path(idx):
    candidates = sorted(VEC_DIR.glob(f"*_{idx:06d}*.vec"))
    if not candidates:
        return None
    return candidates[0]


def process_one_frame(idx, pairs, bg, k):
    vec_path = resolve_vec_path(idx)
    if vec_path is None:
        return None

    I = preprocess(read_pair_mean(pairs, idx), bg)
    C = k * I

    contour_raw, mask_main, thr = detect_boundary_contour(I)
    if contour_raw is None:
        return None

    contour = resample_contour_by_arclength(contour_raw, step_px=BOUNDARY_RESAMPLE_STEP_PX)
    geom = contour_geometry(contour)

    H, W = C.shape
    c_interp = RegularGridInterpolator(
        (np.arange(H, dtype=float), np.arange(W, dtype=float)),
        C,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    c_on_boundary = c_interp(np.column_stack([geom["y"], geom["x"]]))

    vec_meta = parse_vec_header(vec_path)
    x_unique, y_unique, U, V = build_vec_grids(vec_path)
    u_interp = bilinear_interpolator(x_unique, y_unique, U)
    v_interp = bilinear_interpolator(x_unique, y_unique, V)

    bx_mm, by_mm, ds_m = contour_pixels_to_vec_mm(geom, vec_meta)
    pts_mm = np.column_stack([by_mm, bx_mm])

    u_b = u_interp(pts_mm)
    v_b = v_interp(pts_mm)

    valid = np.isfinite(u_b) & np.isfinite(v_b) & np.isfinite(c_on_boundary)
    if np.count_nonzero(valid) < 10:
        return None

    tx = geom["tx"][valid]
    ty = geom["ty"][valid]
    nx = geom["nx"][valid]
    ny = geom["ny"][valid]
    x_px = geom["x"][valid]
    y_px = geom["y"][valid]
    ds_m = ds_m[valid]
    c_b = c_on_boundary[valid]
    u_b = u_b[valid]
    v_b = v_b[valid]

    # convert VEC v (up-positive) to image-like downward-positive for projection
    u_img = u_b
    w_img = -v_b

    w_n = u_img * nx + w_img * ny

    L_valid = np.nansum(ds_m)
    J_cw = np.nansum(c_b * w_n * ds_m)
    Cbar_b = np.nansum(c_b * ds_m) / L_valid
    Wbar_b = np.nansum(w_n * ds_m) / L_valid

    M_total_img = np.nansum(C) * PX_AREA_M2 * DEPTH_M
    mask_area_px = np.count_nonzero(mask_main)
    coverage = np.count_nonzero(valid) / len(contour)

    return {
        "idx": idx,
        "I": I,
        "C": C,
        "thr": thr,
        "contour_full": contour,
        "x_px": x_px,
        "y_px": y_px,
        "nx": nx,
        "ny": ny,
        "vec_meta": vec_meta,
        "x_unique": x_unique,
        "y_unique": y_unique,
        "U": U,
        "V": V,
        "w_n": w_n,
        "c_b": c_b,
        "ds_m": ds_m,
        "L_valid": L_valid,
        "J_cw": J_cw,
        "Cbar_b": Cbar_b,
        "Wbar_b": Wbar_b,
        "M_total_img": M_total_img,
        "mask_area_px": mask_area_px,
        "coverage": coverage,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    idxs, pairs = list_pairs(RAW_DIR)
    if not idxs:
        raise RuntimeError("No LA/LB pairs found in RawData.")

    print(f"Found {len(idxs)} image pairs. Index range: {idxs[0]} .. {idxs[-1]}")

    bg = estimate_background(idxs, pairs, n_frames=N_BG_FRAMES, method=BG_METHOD)
    k = calibrate_k_from_last_frame(idxs, pairs, bg)
    print(f"Calibration k = {k:.6e} [kg/m^3 per normalized intensity unit]")

    results = []
    skipped = []

    for idx in idxs:
        out = process_one_frame(idx, pairs, bg, k)
        if out is None:
            skipped.append(idx)
        else:
            results.append(out)

    if not results:
        raise RuntimeError("No valid frames were processed successfully.")

    if skipped:
        print(f"Skipped {len(skipped)} frames due to missing/invalid contour or VEC overlap.")
        print("Skipped indices:", skipped)

    n = len(results)
    idx_valid = np.array([r["idx"] for r in results], dtype=int)
    time_s = np.arange(n) * dt2_s

    J_cw = np.array([r["J_cw"] for r in results], dtype=float)
    Cbar_b = np.array([r["Cbar_b"] for r in results], dtype=float)
    Wbar_b = np.array([r["Wbar_b"] for r in results], dtype=float)
    L_valid = np.array([r["L_valid"] for r in results], dtype=float)
    M_total_img = np.array([r["M_total_img"] for r in results], dtype=float)
    mask_area_px = np.array([r["mask_area_px"] for r in results], dtype=float)
    coverage = np.array([r["coverage"] for r in results], dtype=float)

    # Auto detect t0 from total mass proxy
    M_norm = M_total_img / np.nanmax(M_total_img)
    if MANUAL_T0_IDX is None:
        t0_idx = int(np.where(M_norm >= MASS_THRESHOLD_FRAC)[0][0])
    else:
        t0_idx = int(MANUAL_T0_IDX)

    # Mean quantities over quasi-steady window
    sl = slice(t0_idx, None)

    J_cw_mean = np.nanmean(J_cw[sl])
    Cbar_mean = np.nanmean(Cbar_b[sl])
    Wbar_mean = np.nanmean(Wbar_b[sl])
    L_mean = np.nanmean(L_valid[sl])

    J_mean_part = np.nanmean(L_valid[sl] * Cbar_b[sl] * Wbar_b[sl])
    J_turb = J_cw_mean - J_mean_part

    # pointwise per-frame turbulent residual using time-mean boundary averages
    J_turb_frame = J_cw - L_valid * Cbar_mean * Wbar_mean
    J_residual_frame = J_cw - L_valid * Cbar_b * Wbar_b

    print("\n===== QUASI-STEADY WINDOW =====")
    print(f"MASS_THRESHOLD_FRAC = {MASS_THRESHOLD_FRAC:.3f}")
    print(f"t0 frame in valid-series = {t0_idx}")
    print(f"t0 physical index = {idx_valid[t0_idx]}")
    print(f"t0 time = {time_s[t0_idx]:.3f} s")

    print("\n===== TIME-AVERAGED RESULTS =====")
    print(f"<J_cw>_t                 = {J_cw_mean:.6e}")
    print(f"<Cbar_b>_t               = {Cbar_mean:.6e} kg/m^3")
    print(f"<Wbar_b>_t               = {Wbar_mean:.6e} m/s")
    print(f"<L_valid>_t              = {L_mean:.6e} m")
    print(f"<L*Cbar_b*Wbar_b>_t      = {J_mean_part:.6e}")
    print(f"Estimated turbulent flux = {J_turb:.6e}")
    print("(using J_turb = <∮C W_n ds> - <L*Cbar_b*Wbar_b>)")

    # ========================================================
    # PLOTS
    # ========================================================
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_s, M_norm, lw=1.8, label="Normalized mass proxy")
    ax1.axvline(time_s[t0_idx], color="red", ls="--", lw=1.2, label="t0")
    ax1.axhline(MASS_THRESHOLD_FRAC, color="gray", ls=":", lw=1.2, label="Threshold")
    ax1.set_title("Mass proxy for quasi-steady window selection")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("M / Mmax [-]")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_s, coverage * 100.0, lw=1.8)
    ax2.axvline(time_s[t0_idx], color="red", ls="--", lw=1.2)
    ax2.set_title("Boundary coverage by valid VEC points")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Coverage [%]")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_s, J_cw, lw=1.6, label="J_cw(t) = ∮ C W_n ds")
    ax3.plot(time_s, L_valid * Cbar_b * Wbar_b, lw=1.3, label="L*C̄_b*W̄_b")
    ax3.axvline(time_s[t0_idx], color="red", ls="--", lw=1.2)
    ax3.set_title("Total transport and mean part")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Flux integral")
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_s, J_residual_frame, lw=1.6, label="J_cw - L*C̄_b(t)*W̄_b(t)")
    ax4.plot(time_s, J_turb_frame, lw=1.3, label="J_cw - L*<C̄_b>_t*<W̄_b>_t")
    ax4.axvline(time_s[t0_idx], color="red", ls="--", lw=1.2)
    ax4.axhline(0, color="gray", ls="--", lw=1.0)
    ax4.set_title("Residual / turbulent-like flux indicators")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Flux residual")
    ax4.legend(loc="best")

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time_s, Cbar_b, lw=1.6, label="C̄_b(t)")
    ax5.plot(time_s, Wbar_b, lw=1.6, label="W̄_b(t)")
    ax5.axvline(time_s[t0_idx], color="red", ls="--", lw=1.2)
    ax5.set_title("Boundary-averaged concentration and normal velocity")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Value")
    ax5.legend(loc="best")

    ax6 = fig.add_subplot(gs[2, 1])
    summary = (
        f"Valid processed frames: {n}\n"
        f"Start index used for averaging: {idx_valid[t0_idx]}\n"
        f"Start time: {time_s[t0_idx]:.3f} s\n"
        f"Mean boundary coverage: {np.nanmean(coverage[sl])*100:.1f}%\n"
        f"Mean valid boundary length: {L_mean:.4e} m\n\n"
        f"<J_cw>_t = {J_cw_mean:.4e}\n"
        f"<L*C̄_b*W̄_b>_t = {J_mean_part:.4e}\n"
        f"Estimated turbulent flux = {J_turb:.4e}\n\n"
        f"Interpretation:\n"
        f"J_turb = <∮ C W_n ds> - <L*C̄_b*W̄_b>\n"
        f"This isolates the transport associated with fluctuations\n"
        f"relative to the time-mean boundary concentration and\n"
        f"time-mean boundary-normal velocity."
    )
    ax6.axis("off")
    ax6.text(0.02, 0.98, summary, va="top", ha="left", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

    fig.suptitle("Time-series estimate of turbulent scalar flux across the droplet boundary", fontsize=15)
    plt.show()

    # Optional example frame visualization
    if SHOW_EXAMPLE_FRAME:
        if EXAMPLE_FRAME_MODE == "t0":
            ex_i = t0_idx
        elif EXAMPLE_FRAME_MODE == "middle":
            ex_i = len(results) // 2
        elif isinstance(EXAMPLE_FRAME_MODE, int):
            ex_i = max(0, min(len(results)-1, EXAMPLE_FRAME_MODE))
        else:
            ex_i = t0_idx

        r = results[ex_i]
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(r["I"], cmap="gray", vmin=0, vmax=1)
        ax.plot(r["contour_full"][:, 1], r["contour_full"][:, 0], color="cyan", lw=1.2)
        ax.scatter(r["x_px"], r["y_px"], s=8, color="yellow")

        mm_per_px_x = r["vec_meta"]["um_per_px_x"] * 1e-3
        mm_per_px_y = r["vec_meta"]["um_per_px_y"] * 1e-3
        Xmm, Ymm = np.meshgrid(r["x_unique"], r["y_unique"])
        Xpx = Xmm / mm_per_px_x + r["vec_meta"]["origin_x_px"]
        Ypx = -Ymm / mm_per_px_y + r["vec_meta"]["origin_y_px"]
        skip = (slice(None, None, QUIVER_SKIP), slice(None, None, QUIVER_SKIP))
        ax.quiver(Xpx[skip], Ypx[skip], r["U"][skip], -r["V"][skip], color="red", scale=None, width=0.002)

        qstep = max(1, len(r["x_px"]) // 40)
        ax.quiver(r["x_px"][::qstep], r["y_px"][::qstep], r["nx"][::qstep], r["ny"][::qstep],
                  color="lime", scale=20, width=0.004)

        ax.set_xlim(0, r["I"].shape[1]-1)
        ax.set_ylim(r["I"].shape[0]-1, 0)
        ax.set_title(
            f"Example frame idx={r['idx']} | coverage={100*r['coverage']:.1f}% | "
            f"J_cw={r['J_cw']:.3e}"
        )
        plt.show()


if __name__ == "__main__":
    main()
