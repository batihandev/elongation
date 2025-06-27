# Subpixel Accuracy Update for Pattern Matching and Marker Detection

## What is Subpixel Accuracy?

Subpixel accuracy means estimating the position of a feature (such as a marker or a matched pattern) with a precision finer than a single pixel. Instead of reporting only integer pixel positions, subpixel methods use mathematical fitting to estimate the true position to a fraction of a pixel (e.g., 123.42 instead of just 123).

## Why is Subpixel Accuracy Useful?

- **Higher precision:** Especially important for measuring small elongations or when high accuracy is required.
- **Reduces quantization error:** Integer-only positions can introduce stepwise errors, especially when the true movement per frame is less than a pixel.
- **More robust measurements:** Subpixel refinement can make your elongation calculations smoother and more reliable.

## Where to Apply in This Project

1. **Pattern Matching (elongation/mark_frames.py):**
   - `match_pattern(strip, pattern, search_range)`
   - `find_pattern_near_reference(strip, pattern, reference_y, max_search_distance=20)`
2. **Marker Detection (elongation/detect_markers.py):**
   - `detect_markers(gray, center_x, ...)` (specifically, the `find_band` function inside it)

## How to Implement Subpixel Refinement

### 1. Add a Helper Function for Parabolic Fitting

Fit a parabola to the matching scores at y-1, y, and y+1 (the best integer position and its neighbors). The vertex of the parabola gives the subpixel minimum.

```python
def subpixel_refine(score_func, y):
    """
    Refine the best integer y position to subpixel accuracy using parabolic fitting.
    score_func: function that returns the matching score at a given y
    y: integer position of best match
    Returns: subpixel y position (float)
    """
    y_vals = np.array([y-1, y, y+1])
    s_vals = np.array([score_func(y-1), score_func(y), score_func(y+1)])
    coeffs = np.polyfit(y_vals, s_vals, 2)
    a, b, c = coeffs
    if a == 0:
        return float(y)  # fallback
    y_subpixel = -b / (2*a)
    return y_subpixel
```

### 2. Update Pattern Matching Functions

- After finding the best integer y (best_y), call `subpixel_refine` with a lambda or function that computes the matching score for nearby y values.
- Use the subpixel y for elongation calculations and overlays.

### 3. Update Marker Detection

- In `find_band`, after detecting a band at y, use the gradient strength at y-1, y, and y+1 to fit a parabola and estimate the subpixel peak.
- Return the subpixel y for the marker position.

## Step-by-Step Instructions

1. **Add the `subpixel_refine` helper function** to a common location (e.g., at the top of `mark_frames.py` and `detect_markers.py`).
2. **In `match_pattern` and `find_pattern_near_reference`:**
   - After finding `best_y`, use `subpixel_refine` to get a subpixel y value.
   - Return and use this value in downstream calculations.
3. **In `detect_markers.py`'s `find_band`:**
   - After finding a band at y, use the gradient strength at y-1, y, and y+1 with `subpixel_refine`.
   - Return the subpixel y for the marker position.
4. **Update any code that uses these positions** to handle float (subpixel) values instead of just integers.
5. **Test the updated code** to ensure elongation measurements are now smoother and more precise.

## Notes

- For patterns or markers near the image edge, ensure y-1 and y+1 are within valid bounds before calling `subpixel_refine`.
- For even more robustness, you can use more points (e.g., y-2 to y+2) and fit a higher-order polynomial, but the 3-point parabola is standard and effective.

---

**This update will improve the precision of your elongation measurements and make your analysis more robust to small changes!**
