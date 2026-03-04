PIV droplet concentration analysis
----------------------------------

This repository contains Python scripts for analyzing PLIF/PIV images of a falling concentration droplet in a water tank.

The goal is to extract concentration fields from the images and analyze transport processes such as concentration flux and temporal concentration fluctuations.

---

# Data

Inside the repository there is a folder named `RawData` containing:

- 50 pairs of PIV images of a falling droplet
- Each time step contains two closely spaced frames (`LA` and `LB`)
- Image format: `.tif`

These images are the raw experimental data used by the analysis scripts.

---

# Files

## `Open_a_TIF.py`

A simple utility script for opening and displaying `.tif` images.

---

## `data_analist_PIV.py`

Performs a **concentration flux analysis** of the droplet.

Main steps:

1. Load image pairs and compute a single intensity field.
2. Remove background and apply filtering.
3. Estimate the droplet boundary from the concentration field.
4. Define a moving boundary line following the droplet.
5. Estimate the concentration flux crossing this boundary.

The computed quantity is a proxy for diffusive flux:

### Flux_proxy ≈ ∫ ( -∇C · n̂ )

---

## `2_point_correlation.py`

Performs a **two-point concentration analysis**.

Main steps:

1. Extract the concentration field from the processed images.
2. Select two spatial points in the flow.
3. Compute concentration time series at both points.
4. Remove the temporal mean to obtain concentration fluctuations:

### c' = c − ⟨c⟩

5. Visualize the evolution of the concentration fluctuations at the two locations.
