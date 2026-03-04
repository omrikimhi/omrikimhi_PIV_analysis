#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================
# Two-point concentration analysis from PLIF experiment 
# ======================================================

"""
Two-point concentration analysis from PLIF images

General workflow:

1. Image pairing
   Each time step contains two images (LA and LB). The images are read and
   averaged to create a single intensity field for that time step.

2. Background removal and filtering
   A background image is estimated from the first frames and subtracted
   from each image. The resulting image is filtered (median + Gaussian)
   and normalized to reduce noise and improve the signal.

3. Concentration calibration
   The total injected volume and density are defined by the user.
   Using the last frame, the image intensity is calibrated so that the
   integrated concentration field corresponds to the known total mass.

4. Point selection
   Two spatial points are selected interactively on the image.
   Around each point a small neighborhood is averaged to obtain a
   representative concentration value.

5. Fluctuation calculation
   For each point the mean concentration is computed over time.
   Concentration fluctuations are then defined as:
       c' = c - <c>

6. Visualization
   The script displays:
   - the processed PLIF image with the two selected points
   - the time series of c' at both points
   - relevant experiment parameters (volume, density, total mass, etc.)
   - a reference line at c' = 0
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
dt2_s  = 0.05  # seconds between indices

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
DEPTH_M = 0.1               # [m] estimation of depth for converting from mass to concentration. This is estimation of squere depth of the volume.

# ===== Setting the Two points =====
USE_MOUSE_PICK = True

P1 = (300, 200) # set default points in case not using mouse pick (x,y) in pixel coordinates.
P2 = (360, 240)

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
V_m3 = INJECTED_VOL_ML * 1e-6
M_total_kg = FLUID_DENSITY * V_m3

I_last = frames_I[-1]
sum_I_last = float(np.sum(I_last))

# Avoid divide by zero
if sum_I_last < 1e-12:
    raise RuntimeError("Sum of last-frame intensity is ~0 after preprocessing. Calibration will fail.")

k = M_total_kg / (sum_I_last * PX_AREA_M2 * DEPTH_M)  # so that C = k*I gives correct total mass

print("Calibration done.")
print(f"M_total_kg = {M_total_kg:.6e} kg, DEPTH_M={DEPTH_M}, k={k:.6e} (kg/m^3 per intensity-unit)")

# ===== Step 4: two-point time series + primes =====

c1 = k * np.array([point_mean(frames_I[i], x1, y1, POINT_RADIUS_PX) for i in range(T)]) # C=K*I Concentration fields (kg/m^3)

c2 = k * np.array([point_mean(frames_I[i], x2, y2, POINT_RADIUS_PX) for i in range(T)])

c1_bar = float(np.mean(c1))
c2_bar = float(np.mean(c2))

c1_p = c1 - c1_bar
c2_p = c2 - c2_bar

# =========================
# VISUALIZATION (animation)
# =========================

# compute fixed y-scale based on max amplitude of the two time series
ymax = np.max(np.abs(np.concatenate([c1_p, c2_p])))
pad = 0.1 * (ymax + 1e-12)

fig = plt.figure(figsize=(12, 5))
ax_img = fig.add_subplot(1, 2, 1)
ax_ts  = fig.add_subplot(1, 2, 2)

# image plot
im = ax_img.imshow(frames_I[0], cmap="gray", vmin=0, vmax=1)
p1_sc = ax_img.scatter([x1], [y1], s=60)
p2_sc = ax_img.scatter([x2], [y2], s=60)

ax_img.set_title("Filtered/normalized droplet + two points")
ax_img.set_xlim(0, W-1)
ax_img.set_ylim(H-1, 0)

# time series plot
line_c1, = ax_ts.plot([], [], linewidth=1.5, label="c1'(t)")
line_c2, = ax_ts.plot([], [], linewidth=1.5, label="c2'(t)")

ax_ts.set_xlim(times.min(), times.max())
ax_ts.set_ylim(-ymax - pad, ymax + pad)

ax_ts.set_xlabel("t [s]")
ax_ts.set_ylabel("C' [kg/m³]")

ax_ts.legend(loc="best")

# zero reference line
ax_ts.axhline(0, color="gray", linewidth=1.5, linestyle="--")

# text box
txt = ax_ts.text(
    0.98, 0.98, "",
    transform=ax_ts.transAxes,
    va="top", ha="right",
    fontsize=11,
    linespacing=1.4,
    bbox=dict(boxstyle="round",
              facecolor="#1f77b4",
              edgecolor="none",
              alpha=0.85)
)

# ANIMATION UPDATE FUNCTION
def update(i):

    # update image
    im.set_data(frames_I[i])

    # update time series
    line_c1.set_data(times[:i+1], c1_p[:i+1])
    line_c2.set_data(times[:i+1], c2_p[:i+1])

    # update text box
    txt.set_text(
        f"V_tot = {V_m3:.2e} [m³]\n"
        f"ρ = {FLUID_DENSITY:.1f} [kg/m³]\n"
        f"M_tot = {M_total_kg:.2e} [kg]\n"
        f"DEPTH = {DEPTH_M:.2g} [m]\n"
        f"<c1> = {c1_bar:.2e}\n"
        f"<c2> = {c2_bar:.2e}\n"
        f"c1' = {c1_p[i]:+.2e}\n"
        f"c2' = {c2_p[i]:+.2e}"
    )

    return im, line_c1, line_c2, txt

# RUN ANIMATION
ani = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=100,
    blit=False
)

plt.tight_layout()
plt.show()