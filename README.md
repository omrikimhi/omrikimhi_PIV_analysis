# PIV droplet concentration analysis
----------------------------------

This repository contains Python scripts for analyzing combined PLIF/PIV measurements of a falling concentration droplet in a water tank.

The project focuses on extracting concentration and velocity information from experimental images in order to investigate turbulent transport, scalar fluxes, and mixing processes near the droplet interface.

---

# Data

Inside the repository there is a folder named `RawData` containing:

- 50 pairs of PLIF/PIV images of a falling droplet
- Each time step contains two closely spaced frames (`LA` and `LB`)
- Image format: `.tif`

The repository also contains an `Analysis` folder with processed PIV vector fields:

- Velocity vector files (`.vec`)
- Each file contains:
  - spatial coordinates
  - velocity components
  - vector validation flags

These datasets are used together to estimate scalar transport and turbulent mixing along the droplet boundary.

---

# Files

## `Open_a_TIF.py`

A simple utility script for opening and displaying `.tif` images.

---

## `Open_VEC.py`

Reads and visualizes Insight `.vec` velocity files.

Main capabilities:

- Read PIV velocity fields
- Remove invalid vectors
- Reconstruct 2D velocity grids
- Display velocity vectors using quiver plots

This script is mainly used for validating the exported PIV data before further analysis.

---

## `data_analist_PIV.py`

Performs a concentration boundary and flux analysis of the droplet.

Main steps:

1. Load image pairs and compute a single intensity field.
2. Remove background and apply filtering.
3. Estimate the droplet boundary from the concentration field.
4. Define a moving boundary following the droplet.
5. Estimate concentration transport across the boundary.

The computed quantity is used as a proxy for diffusive transport:

### Flux_proxy ≈ ∫ ( -∇C · n̂ ) ds

where:

- \(C\) is concentration
- \(n̂\) is the outward boundary normal

The script provides a first estimate of scalar transport across the evolving droplet interface.

---

## `2_point_correlation.py`

Performs a two-point concentration fluctuation analysis from the processed image sequence.

Main steps:

1. Extract the concentration field from the processed images.
2. Select two spatial points in the flow.
3. Compute concentration time series at both points.
4. Remove the temporal mean to obtain concentration fluctuations:

### c' = c − ⟨c⟩

5. Compute the cross-correlation between the two signals:

### Rcc(τ) = ⟨c1'(t) · c2'(t + τ)⟩

6. Identify the correlation peak \(R_{peak}\) and the corresponding time delay \(τ_{peak}\).

### Visualization

The script generates an animated figure showing:

- The filtered/normalized flow image
- The two selected spatial points
- Concentration fluctuations at both locations
- The cross-correlation function

A dashed vertical line indicates the current time frame in the animation.

This analysis is used to estimate transport time scales and spatial coherence within the scalar field.

---

## `boundary_flux.py`

Performs a coupled PLIF/PIV turbulent boundary transport analysis.

Main steps:

1. Extract the concentration field from PLIF image pairs.
2. Detect the instantaneous droplet boundary.
3. Read the PIV velocity field from `.vec` files.
4. Project the velocity onto the local outward boundary normal.
5. Compute local time-averaged concentration and velocity.
6. Compute concentration and velocity fluctuations:

### c' = c − ⟨c⟩

### u_n' = u_n − ⟨u_n⟩

7. Estimate the local turbulent scalar flux:

### u_n' c'

8. Estimate the local turbulent diffusivity:

### D_{t,n} = - (u_n' c') / (∂C/∂n)

9. Integrate quantities along the droplet boundary.

### Visualization

The script generates figures showing:

- Boundary colored by local turbulent flux \(u_n' c'\)
- Boundary colored by local turbulent diffusivity \(D_{t,n}\)
- Flux variation along boundary arc length
- Histogram and statistics of \(D_{t,n}\)

The analysis is used to investigate turbulent scalar transport and mixing near the droplet interface.
