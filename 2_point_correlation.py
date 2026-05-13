#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================
# Two-point concentration analysis from PLIF experiment 
# ======================================================

"""
Two-point concentration analysis from PLIF images

Workflow:

1. Image pairing
   Each time step contains two images (LA and LB). The images are read and
   averaged to create a single intensity field for that time step.

2. Background removal and filtering
   A background image is estimated from the first frames and subtracted.
   The images are then filtered (median + Gaussian) and normalized to
   reduce noise and improve signal quality.

3. Concentration calibration
   The total injected volume and fluid density are defined by the user.
   Using the last frame, image intensity is calibrated so that the
   axisymmetric 3D mass reconstructed from the 2D PLIF plane corresponds
   to the known total injected mass.

4. Point selection
   Two spatial points are selected interactively on the image.
   A small neighborhood around each point is averaged to obtain a
   representative concentration value.

5. Fluctuation calculation
   The temporal mean concentration is computed at each point and
   fluctuations are defined as:
       c' = c - <c>

6. Correlation analysis
   The cross-correlation between the two fluctuation signals is computed:
       Rcc(τ) = <c1'(t) c2'(t + τ)>
   The correlation peak and corresponding time delay are identified.

7. Visualization
   The script displays:
   - the processed PLIF image with the selected points
   - the time series of c1'(t) and c2'(t)
   - the cross-correlation function
   - a moving time indicator synchronized with the animation
"""

# =========================
# IMPORTS
# =========================
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from skimage import io
from skimage.filters import gaussian
from scipy.ndimage import median_filter

# =========================
# SETTINGS
# =========================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR  = BASE_DIR / "RawData"

# Timing
dt2_s  = 1  # seconds between indices (information from tstmp in rowdata)

# Spatial scaling
PX_PER_MM = 16.0
M_PER_PX  = 1e-3 / PX_PER_MM
PX_AREA_M2 = M_PER_PX**2

# Preprocessing
N_BG_FRAMES = 5
BG_METHOD   = "median"  # "median" or "mean"
MEDIAN_SIZE = 3
SMOOTH_SIGMA = 1.0

# ===== Definition of the injected material =====
INJECTED_VOL_ML = 50.0      # [mL]
FLUID_DENSITY = 1000.0      # [kg/m^3]

'''
Concentration calibration assumption is cylindrical symmetry around a vertical axis.
 Axis options:
   "centroid"     - x-axis from intensity-weighted centroid of the last frame
   "frame_center" - x-axis at the center of the image
   "manual"       - use CYL_MANUAL_AXIS_X_PX
'''
CYL_AXIS_MODE = "centroid"
CYL_MANUAL_AXIS_X_PX = None  # e.g. 1500.0, used only if CYL_AXIS_MODE = "manual"
CYL_R_EPS_PX = 1e-6          # avoids zero radius exactly on the symmetry axis

# ===== Setting the Two points =====
USE_MOUSE_PICK = False

P1 = (1500, 600) # set default points in case not using mouse pick (x,y) in pixel coordinates.
P2 = (1500, 800)

POINT_RADIUS_PX = 3   # 0 = single pixel, 2 = (5x5) mean, etc.

# =========================
# Helpers: file parsing
# =========================

def list_pairs(raw_dir):
    files = [p for p in raw_dir.iterdir()
             if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]

    pairs = {}  # idx -> {"LA": Path, "LB": Path}

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


def estimate_background(idxs, pairs, n_frames=10, method="median"):
    n = min(n_frames, len(idxs))
    stack = []
    for i in range(n):
        stack.append(read_pair_mean(pairs, idxs[i]))
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

    # robust-ish normalization to [0,1]
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.5)
    x = (x - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0, 1).astype(np.float32)

    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        x = gaussian(x, sigma=SMOOTH_SIGMA, preserve_range=True).astype(np.float32)

    return x


def point_mean(img, x, y, r=0):
    x = int(round(x))
    y = int(round(y))
    x0 = max(0, x - r); x1 = min(W, x + r + 1)
    y0 = max(0, y - r); y1 = min(H, y + r + 1)
    patch = img[y0:y1, x0:x1]
    return float(np.mean(patch))


def get_cyl_axis_x(I_last):
    """Return the x-location of the cylindrical symmetry axis in pixels."""
    H_local, W_local = I_last.shape

    if CYL_AXIS_MODE == "frame_center":
        return 0.5 * (W_local - 1)

    if CYL_AXIS_MODE == "manual":
        if CYL_MANUAL_AXIS_X_PX is None:
            raise ValueError("CYL_MANUAL_AXIS_X_PX must be set when CYL_AXIS_MODE='manual'.")
        return float(CYL_MANUAL_AXIS_X_PX)

    if CYL_AXIS_MODE == "centroid":
        weights = np.asarray(I_last, dtype=float)
        total = np.nansum(weights)
        if not np.isfinite(total) or total <= 1e-12:
            raise RuntimeError("Cannot compute centroid axis: last-frame intensity sum is too small.")

        x_coords = np.arange(W_local, dtype=float)
        x_weighted = np.nansum(weights * x_coords[None, :])
        return float(x_weighted / total)

    raise ValueError("CYL_AXIS_MODE must be 'centroid', 'frame_center', or 'manual'.")


def cylindrical_full_cross_section_weights(I_shape, axis_x_px):
    """
    Volume weight for each pixel under cylindrical symmetry.

    The image is assumed to show the full left+right cross-section of the flow.
    Therefore each radial shell appears twice in the image, so the shell factor
    is pi*r instead of 2*pi*r.

    dV_pixel = pixel_area * pi * r
    where r is the horizontal distance from the symmetry axis.
    """
    H_local, W_local = I_shape
    x_coords = np.arange(W_local, dtype=float)
    r_px = np.abs(x_coords - axis_x_px)
    r_m = np.maximum(r_px * M_PER_PX, CYL_R_EPS_PX * M_PER_PX)
    return (np.pi * r_m[None, :] * PX_AREA_M2).astype(np.float64)


def calibrate_k_cylindrical(I_last):
    """Calibrate intensity-to-concentration using cylindrical symmetry."""
    V_m3 = INJECTED_VOL_ML * 1e-6
    M_total_kg = FLUID_DENSITY * V_m3

    axis_x_px = get_cyl_axis_x(I_last)
    dV_weights = cylindrical_full_cross_section_weights(I_last.shape, axis_x_px)

    weighted_sum = float(np.nansum(I_last * dV_weights))
    if weighted_sum < 1e-12:
        raise RuntimeError("Weighted last-frame intensity is ~0 after preprocessing. Calibration will fail.")

    k = M_total_kg / weighted_sum
    return k, axis_x_px, M_total_kg, V_m3

# =========================
# MAIN
# =========================

idxs, pairs = list_pairs(RAW_DIR)
if len(idxs) == 0:
    raise RuntimeError("No LA/LB tif pairs found. Check RAW_DIR and filenames.")
print(f"Found {len(idxs)} complete LA/LB pairs.")

bg = estimate_background(idxs, pairs, n_frames=N_BG_FRAMES, method=BG_METHOD)
print("Background estimated.")

# Load frames (normalized intensity)
frames_I = []
for idx in idxs:
    I = preprocess(read_pair_mean(pairs, idx), bg)
    frames_I.append(I)
frames_I = np.stack(frames_I, axis=0)  # (T,H,W)
T, H, W = frames_I.shape
times = np.arange(T) * dt2_s

# ===== Pick two points visually (operate if "USE_MOUSE_PICK = True" in setting) =====
if USE_MOUSE_PICK:
    fig_pick, ax_pick = plt.subplots(figsize=(7, 5))
    ax_pick.imshow(frames_I[-1], cmap="gray", vmin=0, vmax=1)
    ax_pick.set_title("Click TWO points (P1 then P2), then press Enter")
    pts = plt.ginput(2, timeout=-1)  # returns [(x,y), (x,y)]
    plt.close(fig_pick)

    (x1, y1), (x2, y2) = pts
    P1 = (int(round(x1)), int(round(y1)))
    P2 = (int(round(x2)), int(round(y2)))
    print("Picked:", P1, P2)
else:
    x1, y1 = P1
    x2, y2 = P2

# ===== Step 3: calibration =====
I_last = frames_I[-1]

k, cyl_axis_x_px, M_total_kg, V_m3 = calibrate_k_cylindrical(I_last)

print("Calibration done using cylindrical symmetry.")
print(f"M_total_kg = {M_total_kg:.3e} kg")
print(f"CYL_AXIS_MODE = {CYL_AXIS_MODE}, axis_x = {cyl_axis_x_px:.2f} px")
print(f"k = {k:.3e} (kg/m^3 per intensity-unit)")

# ===== Step 4: two-point time series + primes =====

c1 = k * np.array([point_mean(frames_I[i], x1, y1, POINT_RADIUS_PX) for i in range(T)]) # C=K*I Concentration fields (kg/m^3)

c2 = k * np.array([point_mean(frames_I[i], x2, y2, POINT_RADIUS_PX) for i in range(T)])

c1_bar = float(np.mean(c1))
c2_bar = float(np.mean(c2))

c1_p = c1 - c1_bar
c2_p = c2 - c2_bar

# =========================
# CROSS-CORRELATION (static)
# =========================

# Cross-correlation R_cc(tau) = < c1'(t)*c2'(t+tau) >
# We'll compute the normalized correlation coefficient so values are in [-1, 1].

c1p = c1_p - np.mean(c1_p)
c2p = c2_p - np.mean(c2_p)

N = len(c1p)
dt = dt2_s

# raw cross-correlation (full lags)
R_raw = np.correlate(c1p, c2p, mode="full")  # length 2N-1
lags = np.arange(-(N-1), N)                  # sample lags
tau = lags * dt                              # time lags [s]

# normalize to correlation coefficient (avoid divide by zero)
den = (np.std(c1p) * np.std(c2p) * N) + 1e-12
R = R_raw / den  # now roughly in [-1, 1]

# "c1 leads c2" (tau >= 0):
mask_pos = tau >= 0
tau_pos = tau[mask_pos]
R_pos = R[mask_pos]

# peak (for tau >= 0)
i_peak = int(np.argmax(R_pos))
tau_peak = float(tau_pos[i_peak])
R_peak = float(R_pos[i_peak])

# vertical separation and apparent downward speed
dy_px = abs(y2 - y1)
dy_m = dy_px * M_PER_PX   # positive downward if image coordinates increase downward

if abs(tau_peak) > 1e-12:
    Vy_app = dy_m / tau_peak
else:
    Vy_app = np.nan

# =========================
# VISUALIZATION (animation)
# =========================

# compute fixed y-scale based on max amplitude of the two time series
ymax = float(np.max(np.abs(np.concatenate([c1_p, c2_p]))))
pad = 0.1 * (ymax + 1e-12)

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(
    nrows=2, ncols=3,
    height_ratios=[14, 0.9],   # main row for plots, bottom row for text
    hspace=0.18, wspace=0.28
)

ax_img  = fig.add_subplot(gs[0, 0])
ax_ts   = fig.add_subplot(gs[0, 1])
ax_corr = fig.add_subplot(gs[0, 2])

ax_txt  = fig.add_subplot(gs[1, :])  # text box spanning the bottom row
ax_txt.axis("off")

# image plot
im = ax_img.imshow(frames_I[0], cmap="gray", vmin=0, vmax=1)
p1_sc = ax_img.scatter([x1], [y1], s=60)
p2_sc = ax_img.scatter([x2], [y2], s=60)

ax_img.set_title("Droplet animation + the two points")
ax_img.set_xlim(0, W-1)
ax_img.set_ylim(H-1, 0)

# time series plot
line_c1, = ax_ts.plot([], [], linewidth=1.5, label="c1'(t)")
line_c2, = ax_ts.plot([], [], linewidth=1.5, label="c2'(t)")

ax_ts.set_xlim(times.min(), times.max())
ax_ts.set_ylim(-ymax - pad, ymax + pad)

ax_ts.set_xlabel("t [s]")
ax_ts.set_ylabel("C' [kg/m³]")
ax_ts.set_title("Concentration fluctuations at two points")

ax_ts.legend(loc="best")

# zero reference line
ax_ts.axhline(0, color="gray", linewidth=1.5, linestyle="--")

# vertical line showing current frame time
time_line, = ax_ts.plot([times[0], times[0]], [ax_ts.get_ylim()[0], ax_ts.get_ylim()[1]],
                        linestyle="--", linewidth=1, color="red", label="Current time")
# --- correlation plot (static) ---
line_corr, = ax_corr.plot(tau_pos, R_pos, linewidth=1.5, label="R_cc(τ)")
ax_corr.set_xlabel("τ [s]")
ax_corr.set_ylabel("R_cc(τ)")
ax_corr.set_title("Cross-correlation\nR_cc(τ) = < c1'(t)*c2'(t+τ)>")

# reference lines
ax_corr.axhline(0, color="gray", linewidth=1.0, linestyle="--")
vline_peak = ax_corr.axvline(tau_peak, color="gray", linewidth=1.0, linestyle="--")

# nice limits
ax_corr.set_xlim(tau_pos.min(), tau_pos.max())
ax_corr.set_ylim(-1.05, 1.05)

# small annotation
corr_txt = ax_corr.text(
    0.98, 0.98,
    f"τ_peak = {tau_peak:.3g} [s]\n"
    f"R_peak = {R_peak:.3g}\n"
    f"Δy = {dy_m*1e3:.4g} [mm]\n"
    f"V_y = {Vy_app*1e3:.4g} [mm/s]",
    transform=ax_corr.transAxes,
    va="top", ha="right",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8)
)

# text box
txt = ax_txt.text(
    0.5, 0.18, "",
    ha="center", va="center",
    fontsize=11,
    color="white",
    linespacing=1.3,
    bbox=dict(boxstyle="round", facecolor="#1f77b4", edgecolor="none", alpha=0.9),
    transform=ax_txt.transAxes
)

# ANIMATION UPDATE FUNCTION
def update(i):

    # update image
    im.set_data(frames_I[i])

    # update time series
    line_c1.set_data(times, c1_p)
    line_c2.set_data(times, c2_p)

    # update moving time indicator
    time_line.set_xdata([times[i], times[i]])

    # update text box
    txt.set_text(
        f"V_tot = {V_m3:.2e} [m³] | ρ = {FLUID_DENSITY:.1f} [kg/m³] | M_tot = {M_total_kg:.2e} [kg] | axis x = {cyl_axis_x_px:.1f} px\n"
        f"<c1> = {c1_bar:.2e} | c1' = {c1_p[i]:+.2e} | <c2> = {c2_bar:.2e} | c2' = {c2_p[i]:+.2e}"
    )
    return im, line_c1, line_c2, time_line, txt


def save_single_axis(ax, filename, expand=(1.02, 1.02)):
    """Save one axis from the combined figure as a separate PDF."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(*expand)
    fig.savefig(filename, bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))

ax_img.set_box_aspect(1)
ax_ts.set_box_aspect(1)
ax_corr.set_box_aspect(1)

fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.06)
update(T - 1)
#save_single_axis(ax_img, "2_point_correlation_image_last_frame.pdf")
#save_single_axis(ax_ts, "2_point_correlation_timeseries_last_frame.pdf")
#save_single_axis(ax_corr, "2_point_correlation_correlation_last_frame.pdf")

# RUN ANIMATION
ani = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=100,
    blit=False
)
plt.show()