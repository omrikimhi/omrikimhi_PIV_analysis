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
   with a synthetic mapping C = I_norm * C_MAX (until you have calibration).
   If you know a diffusion coefficient D, set D_DIFF to compute ∫(-D∇C·n̂)ds.
7) Outputs: animation (image + contour + flux time series) and optional GIF saved
   OUTSIDE RawData (in BASE_DIR or an output folder).

Notes
- Without calibration (and without D), treat flux as a trend/proxy, not an absolute value.
