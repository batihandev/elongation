# Heavy Processing Parameters Documentation

This document explains the enhanced parameters used in the heavy processing targets (`sample1_heavy` and `sample2_heavy`) for better marking quality and more robust pattern detection.

## Overview

The heavy processing targets use `every_n_frames=1` (extract all frames) combined with optimized pattern detection parameters to achieve the highest quality results, at the cost of longer processing time.

## Parameter Changes from Default

| Parameter                | Default | Heavy Processing | Rationale                                          |
| ------------------------ | ------- | ---------------- | -------------------------------------------------- |
| `every_n_frames`         | 5       | 1                | Extract all frames for maximum temporal resolution |
| `skip_start_seconds`     | 0.0     | 2.0              | Skip first 2 seconds to avoid initial camera setup |
| `auto_detect_focus`      | True    | True             | Automatically detect and skip unfocused frames     |
| `auto_detect_motion`     | True    | True             | Automatically detect and skip unstable frames      |
| `focus_threshold`        | 50      | 50               | Laplacian variance threshold for focus detection   |
| `motion_threshold`       | 1000    | 1000             | Optical flow threshold for motion detection        |
| `min_consecutive_stable` | 15      | 15               | Minimum consecutive stable frames required         |
| `max_pattern_height`     | 12      | 15               | Larger patterns capture more texture information   |
| `pattern_width`          | 8       | 10               | Wider patterns include more horizontal texture     |
| `search_margin_y`        | 5       | 8                | Larger vertical search area for pattern matching   |
| `search_margin_x`        | 5       | 8                | Larger horizontal search area around rebar center  |
| `pattern_capture_frames` | 10      | 15               | More frames analyzed for better pattern selection  |
| `pattern_capture_step`   | 5       | 3                | Smaller step for more thorough pattern analysis    |
| `top_n_to_keep`          | 5       | 8                | Keep more pattern candidates for better selection  |
| `prune_threshold`        | 1e7     | 5e6              | Stricter filtering for higher quality matches      |
| `scan_width`             | 10      | 12               | Wider strip for more robust rebar center detection |
| `threshold_ratio`        | 0.3     | 0.25             | More sensitive gradient detection                  |
| `min_valid_distance`     | 50      | 60               | Require more separation for reliable gauge length  |
| `max_band_thickness`     | 10      | 12               | Allow slightly thicker texture bands               |

## Usage

### Command Line

```bash
# Process sample1 with heavy parameters
make sample1_heavy

# Process sample2 with heavy parameters
make sample2_heavy
```

### Expected Processing Time

- **Standard processing** (`sample1`, `sample2`): ~2-5 minutes
- **Heavy processing** (`sample1_heavy`, `sample2_heavy`): ~10-20 minutes

The heavy processing takes significantly longer because:

1. **More frames**: `every_n_frames=1` vs `every_n_frames=5` (5x more frames)
2. **Larger search areas**: Increased search margins require more computation
3. **More pattern analysis**: More frames analyzed for pattern selection
4. **Stricter filtering**: Lower prune threshold requires more computation

## Quality Improvements

### Expected Benefits

1. **Higher temporal resolution**: All frames captured for smoother elongation curves
2. **Better pattern matching**: Larger search areas and patterns improve robustness
3. **More reliable detection**: Stricter filtering reduces false positives
4. **Improved accuracy**: More thorough pattern selection from more frames
5. **Automatic focus detection**: Skips unfocused frames automatically for better quality
6. **Flexible skipping**: Both frame-based and time-based skipping options

### Focus Detection Benefits

- **Automatic quality control**: No need to manually identify unfocused frames
- **Consistent results**: Ensures all processed frames are in focus
- **Time savings**: Automatically skips camera setup and focus adjustment periods
- **Logging**: Detailed focus detection logs for analysis and debugging
- **Configurable sensitivity**: Adjustable threshold for different video qualities

### Motion Detection Benefits

- **Camera shake detection**: Automatically identifies and skips frames with excessive movement
- **Stability assurance**: Ensures all processed frames are from stable camera periods
- **Movement filtering**: Removes frames affected by camera adjustments, vibrations, or operator movement
- **Optical flow analysis**: Uses advanced computer vision to detect subtle motion patterns
- **Comprehensive coverage**: Grid-based sampling ensures motion detection across the entire frame

### Combined Stability Detection

- **Dual validation**: Frames must pass both focus AND motion quality checks
- **Consecutive stability**: Requires multiple stable frames to ensure consistent quality
- **Automatic logging**: Detailed stability logs with both focus and motion scores
- **Flexible configuration**: Can enable/disable focus and motion detection independently

### When to Use Heavy Processing

- **Research applications**: When maximum accuracy is required
- **Problematic videos**: When standard processing produces poor results
- **High-speed tests**: When temporal resolution is critical
- **Quality validation**: To verify results from standard processing

### When to Use Standard Processing

- **Routine testing**: For regular quality control
- **Batch processing**: When processing multiple videos
- **Quick results**: When speed is more important than maximum accuracy
- **Initial analysis**: For preliminary results before heavy processing

## Parameter Tuning Guidelines

### For Even Higher Quality (Very Slow)

```python
# Ultra-high quality settings
every_n_frames=1
max_pattern_height=18
pattern_width=12
search_margin_y=10
search_margin_x=10
pattern_capture_frames=20
pattern_capture_step=2
top_n_to_keep=10
prune_threshold=2e6
```

### For Balanced Quality/Speed

```python
# Balanced settings (current heavy processing)
every_n_frames=1
max_pattern_height=15
pattern_width=10
search_margin_y=8
search_margin_x=8
pattern_capture_frames=15
pattern_capture_step=3
top_n_to_keep=8
prune_threshold=5e6
```

### For Faster Processing

```python
# Faster settings
every_n_frames=2
max_pattern_height=12
pattern_width=8
search_margin_y=6
search_margin_x=6
pattern_capture_frames=10
pattern_capture_step=5
top_n_to_keep=5
prune_threshold=1e7
```

## Troubleshooting

### If Heavy Processing Fails

1. **Check disk space**: Heavy processing generates many more files
2. **Monitor memory usage**: Larger parameters require more RAM
3. **Verify video quality**: Heavy processing is more sensitive to poor video quality
4. **Check parameter compatibility**: Some parameter combinations may not work well together

### Performance Optimization

1. **Use SSD storage**: Faster I/O for frame extraction
2. **Increase RAM**: More memory for pattern analysis
3. **Close other applications**: Free up CPU resources
4. **Monitor system resources**: Use `htop` or similar tools

## Integration with Backend

The heavy processing parameters can be passed to the FastAPI backend:

```bash
curl -X POST http://localhost:8000/process/ \
  -F "video=@sample/40kn-1.mp4" \
  -F "every_n_frames=1" \
  -F "max_pattern_height=15" \
  -F "pattern_width=10" \
  -F "search_margin_y=8" \
  -F "search_margin_x=8" \
  -F "pattern_capture_frames=15" \
  -F "pattern_capture_step=3" \
  -F "top_n_to_keep=8" \
  -F "prune_threshold=5000000" \
  -F "scan_width=12" \
  -F "threshold_ratio=0.25" \
  -F "min_valid_distance=60" \
  -F "max_band_thickness=12"
```

---

_This documentation should be updated when parameters are modified or new optimization strategies are discovered._
