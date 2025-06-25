# Marker Detection: Current Approach and Suggestions for Improvement

## How Marker Detection Works

### 1. **Detection Algorithm**

- The function `detect_markers` (in `elongation/detect_markers.py`) is responsible for finding the top and bottom gauge markers on the rebar in each frame.
- **Steps:**
  1. **Region Extraction:** A vertical strip centered at the detected rebar center (`center_x`) is extracted from the grayscale image.
  2. **Contrast Enhancement:** Histogram equalization is applied to the strip to improve contrast.
  3. **Edge Detection:** A vertical Sobel filter is used to highlight horizontal transitions (marker edges).
  4. **Gradient Analysis:** The sum of absolute vertical gradients is computed for each row, producing a 1D profile (`gradient_strength`).
  5. **Thresholding:** Rows where the gradient exceeds a fraction (`threshold_ratio`) of the maximum are considered marker candidates.
  6. **Band Search:** The code searches for two strong bands (top and bottom markers) in the upper and lower middle sections of the strip, respectively.
  7. **Validation:** If the detected bands are too close together (`min_valid_distance`), the detection is rejected.

### 2. **Pattern Matching (in `mark_frames.py`)**

- After initial marker detection, small image patches (patterns) are extracted around the detected marker locations in early frames.
- For subsequent frames, these patterns are matched using template matching to improve robustness against noise and lighting changes.

### 3. **(Optional) Stability Enhancement**

- In some versions (e.g., `old/mark_frames_v6.py`), a function called `stable_marker_detection` runs the marker detection multiple times with small random jitters in the center position, then takes the median result and checks the standard deviation. If the results are unstable (high std), the frame is flagged as unreliable.

---

## Why Inconsistencies Occur

- **Lighting and Contrast:** If the markers are faint or the lighting changes, the Sobel/gradient method may miss or misplace the bands.
- **Rebar Texture:** The rebar's own texture or surface defects can create false positives in the gradient profile.
- **Camera Movement:** Even small shifts in the rebar's position or camera shake can throw off the center and thus the marker search.
- **Parameter Sensitivity:** The method is sensitive to `scan_width`, `threshold_ratio`, and other parameters, which may not generalize across all frames or videos.
- **Noisy Background:** If the background has strong horizontal features, it can interfere with marker detection.

---

## Suggestions for More Robust Marker Detection

### 1. **Use Stability/Consensus Methods**

- **Jittered Consensus:** As in `stable_marker_detection`, run detection multiple times with small jitters in `center_x` and use the median result. Discard frames with high standard deviation.
- **Temporal Smoothing:** Track marker positions over time and reject outliers or sudden jumps.

### 2. **Improve Preprocessing**

- **Adaptive Histogram Equalization:** Use CLAHE instead of global equalization for better local contrast.
- **Denoising:** Apply a denoising filter (e.g., bilateral filter) before edge detection.

### 3. **Enhance Edge Detection**

- **Canny Edge Detector:** Try Canny instead of Sobel for sharper edge localization.
- **Multi-scale Gradients:** Combine results from different Sobel kernel sizes.

### 4. **Template/Pattern Matching**

- **Patch Matching:** After initial detection, always use template matching with the extracted marker patches for subsequent frames.
- **Pattern Update:** Periodically update the reference patterns if lighting or appearance changes.

### 5. **Machine Learning Approaches**

- **Classical ML:** Train a simple classifier (e.g., SVM) on small patches to distinguish marker vs. non-marker.
- **Deep Learning:** For best results, train a small CNN to localize markers, if you have enough labeled data.

### 6. **Parameter Auto-tuning**

- **Dynamic Thresholds:** Adjust `threshold_ratio` based on image statistics or previous detection confidence.
- **Scan Range Adaptation:** If detection fails, expand the search region or try multiple parameter sets.

### 7. **Background Suppression**

- **Masking:** Use a mask to ignore background regions outside the rebar strip.
- **Background Subtraction:** If the background is static, subtract a background model to enhance marker contrast.

---

## Example: Robust Marker Detection (Pseudocode)

```python
def robust_marker_detection(gray, center_x, ...):
    results = []
    for dx in range(-2, 3):
        top, bottom = detect_markers(gray, center_x + dx, ...)
        if top is not None and bottom is not None:
            results.append((top, bottom))
    if len(results) < 3:
        return None, None  # Not enough consensus
    tops, bottoms = zip(*results)
    if np.std(tops) > 5 or np.std(bottoms) > 5:
        return None, None  # Too unstable
    return int(np.median(tops)), int(np.median(bottoms))
```

---

## References in Codebase

- `elongation/detect_markers.py`: Main marker detection logic.
- `old/mark_frames_v6.py`: Example of robust, consensus-based detection.
- `elongation/mark_frames.py`: Pattern matching and frame processing pipeline.

---

## Summary

The current marker detection is based on vertical gradient analysis and is sensitive to noise, lighting, and parameter choices. For better consistency:

- Use consensus/jittered detection and temporal smoothing.
- Improve preprocessing and edge detection.
- Rely more on template/pattern matching after initial detection.
- Consider ML-based approaches for further robustness.
