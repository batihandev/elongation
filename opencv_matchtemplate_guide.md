# OpenCV matchTemplate Implementation Guide

## Overview

Replace the current manual sliding window pattern matching with OpenCV's highly optimized `cv2.matchTemplate` function. This will significantly improve performance while maintaining subpixel accuracy.

## Key Changes Required

### 1. Replace `match_pattern` Function

**Current approach:** Manual sliding window with normalization and denoising for each candidate window.

**New approach:** Use `cv2.matchTemplate` with normalized correlation coefficient method.

```python
def match_pattern(strip, pattern, search_range):
    # Restrict the strip to the vertical search range
    y1, y2 = search_range
    y2 = min(y2, strip.shape[0])
    search_strip = strip[y1:y2, :]
    if search_strip.shape[0] < pattern.shape[0]:
        return None, None

    # Use matchTemplate (single channel, so input must be float32)
    search_strip_f = search_strip.astype(np.float32)
    pattern_f = pattern.astype(np.float32)
    res = cv2.matchTemplate(search_strip_f, pattern_f, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    best_y_local = max_loc[1]  # y in the local search_strip
    best_y = y1 + best_y_local

    # For subpixel: fit a parabola to the best_y and its neighbors
    if 0 < best_y_local < res.shape[0] - 1:
        y_vals = np.array([best_y_local - 1, best_y_local, best_y_local + 1])
        scores = res[y_vals, 0]
        # Parabola fit: vertex = -b/(2a)
        denom = 2 * (scores[0] - 2 * scores[1] + scores[2])
        if denom != 0:
            delta = (scores[0] - scores[2]) / denom
            best_y_subpixel = best_y + delta
        else:
            best_y_subpixel = float(best_y)
    else:
        best_y_subpixel = float(best_y)

    # For TM_CCOEFF_NORMED, higher is better (max_val)
    best_score = max_val
    return best_y_subpixel, best_score
```

### 2. Update `find_pattern_near_reference` Function

**Current approach:** Loop through distance offsets and test each position.

**New approach:** Extract a search window around the reference position and use `matchTemplate`.

```python
def find_pattern_near_reference(strip, pattern, reference_y, max_search_distance=20):
    h = strip.shape[0]

    # Define search window around reference_y
    y1 = max(0, reference_y - max_search_distance)
    y2 = min(h, reference_y + max_search_distance + pattern.shape[0])
    search_strip = strip[y1:y2, :]

    if search_strip.shape[0] < pattern.shape[0]:
        return None, None

    # Use matchTemplate
    search_strip_f = search_strip.astype(np.float32)
    pattern_f = pattern.astype(np.float32)
    res = cv2.matchTemplate(search_strip_f, pattern_f, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    best_y_local = max_loc[1]  # y in the local search_strip
    best_y = y1 + best_y_local

    # Subpixel refinement
    if 0 < best_y_local < res.shape[0] - 1:
        y_vals = np.array([best_y_local - 1, best_y_local, best_y_local + 1])
        scores = res[y_vals, 0]
        denom = 2 * (scores[0] - 2 * scores[1] + scores[2])
        if denom != 0:
            delta = (scores[0] - scores[2]) / denom
            best_y_subpixel = best_y + delta
        else:
            best_y_subpixel = float(best_y)
    else:
        best_y_subpixel = float(best_y)

    best_score = max_val
    return best_y_subpixel, best_score
```

### 3. Remove Custom Normalization/Denoising

**Remove this function entirely:**

```python
def normalize_and_denoise(img):
    # Apply Gaussian blur for denoising
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # Normalize to zero mean, unit variance
    img_norm = img_blur.astype(float)
    img_norm = (img_norm - np.mean(img_norm)) / (np.std(img_norm) + 1e-8)
    return img_norm
```

**Why remove it?** OpenCV's `cv2.TM_CCOEFF_NORMED` does normalization internally and much more efficiently. The current code applies normalization to every single candidate window during sliding window search, which is extremely expensive. OpenCV does this optimization automatically.

### 4. Update Score Logic

**Important:** With `cv2.TM_CCOEFF_NORMED`:

- **Higher scores are better** (closer to 1.0 = perfect match)
- **Lower scores are worse** (closer to -1.0 = anti-correlation)

**Update these sections:**

#### In `filter_consistent_pairs`:

```python
# Change from: if score < best_score:
# To:
if score > best_score:  # Higher is better for TM_CCOEFF_NORMED
    best_score = score
    best_y = y
```

#### In `score_candidate_pair`:

```python
# Change from: if score < best_score:
# To:
if score > best_score:  # Higher is better for TM_CCOEFF_NORMED
    best_score = score
    best_pattern = (pt, pb)
    match_results = temp_results
```

### 5. Alternative: Use TM_SQDIFF_NORMED

If you prefer squared difference (lower is better):

```python
# In match_pattern and find_pattern_near_reference:
res = cv2.matchTemplate(search_strip_f, pattern_f, cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
best_y_local = min_loc[1]  # Use min_loc instead of max_loc
best_score = min_val  # Use min_val instead of max_val
```

## Benefits

1. **Performance:** 10-100x faster than manual sliding window
2. **Optimization:** OpenCV uses highly optimized SIMD instructions
3. **Robustness:** Built-in normalization handles lighting variations
4. **Accuracy:** Maintains subpixel precision with parabolic fitting

## Implementation Notes

- **Input types:** Convert to `float32` for best results
- **Score interpretation:** `TM_CCOEFF_NORMED` returns values in [-1, 1] range
- **Subpixel accuracy:** Still works with parabolic fitting on the correlation surface
- **Memory usage:** More efficient than storing scores for all positions
- **No manual normalization needed:** OpenCV handles this internally and efficiently

## Testing

After implementation, verify:

1. Pattern matching accuracy is maintained
2. Subpixel precision is preserved
3. Performance improvement is significant
4. Score thresholds may need adjustment due to different scale

## Files to Modify

- `elongation/mark_frames.py` - Main file containing the pattern matching functions
- Remove `normalize_and_denoise` function calls from matching loops
- Update score comparison logic throughout the file
