# Elongation-Rebar

## Project Overview

**Elongation-Rebar** is a Python-based system for analyzing the elongation of rebar (reinforcing steel bars) during mechanical testing, using video footage. The project extracts frames from test videos, detects and tracks natural texture patterns on the rebar (rather than relying on paint marks), and computes elongation over time. The analysis includes:

- Frame extraction from video
- Pattern extraction and matching using the rebar's natural texture (focused on grid lines, e.g., 30% and 70% of the image height)
- Visualization with fixed reference (yellow) and current (blue) markers
- Elongation calculation and smoothing
- Pixel-to-mm conversion and force estimation
- Batch processing and PDF report generation
- FastAPI backend for web-based interaction

## Project Structure

- `elongation/` — Core image processing, pattern extraction, analysis, and reporting modules
- `backend/` — FastAPI backend for file upload, processing, and API endpoints
- `frontend/` — (Nuxt-based) web frontend
- `sample/` — Sample test videos
- `results/` — Output data, plots, and reports
- `process_video.py` — CLI entry point for single video processing

## Key Features

- **Pattern Extraction:** Uses the natural texture of the rebar, extracting patterns at fixed grid lines for robust, paint-free tracking.
- **Reference and Current Markers:** Yellow lines (reference) are fixed at the best pattern locations from the first frame; blue lines (current) are drawn at matched locations in each frame.
- **Batch and Single Processing:** Supports both batch automation and single video analysis.
- **Pixel-to-mm Conversion:** Allows calibration for real-world measurements.
- **Force Curve Estimation:** Computes estimated force using material properties.
- **PDF Reporting:** Generates summary reports with plots.

## Usage

### Command Line

```bash
python process_video.py --video <video_path> [options]
```

### FastAPI Backend

See `backend/main.py` for API endpoints. Example usage:

```bash
curl -X POST http://localhost:8000/process/ \
  -F "video=@sample/40kn-1.mp4" \
  -F "every_n_frames=1" \
  -F "pattern_height=15" \
  -F "pattern_width=15" \
  -F "pattern_top_grid=7" \
  -F "pattern_bottom_grid=3" \
  -F "num_divisions=75" \
  -F "min_gap=25" \
  -F "dbscan_eps=0.5"
```

### Pattern Detection Parameters (`process_images`)

Current parameters (from `elongation/mark_frames.py`):

- `pattern_height` (default: 15): Height of pattern windows to extract in pixels
- `pattern_width` (default: 15): Width of pattern windows to extract in pixels
- `pattern_top_grid` (default: 7): Grid line (0=bottom, 10=top) for top pattern extraction
- `pattern_bottom_grid` (default: 3): Grid line (0=bottom, 10=top) for bottom pattern extraction
- `num_divisions` (default: 75): Number of grid points (patterns) to extract between top and bottom
- `min_gap` (default: 25): Minimum gap (in grid indices) between patterns for measurement
- `dbscan_eps` (default: 0.5): DBSCAN clustering range for elongation grouping
- `min_contrast` (default: 10): Minimum contrast for pattern validation
- `min_edge_strength` (default: 5): Minimum edge strength for pattern validation
- `fallback_search_radius` (default: 50): Search radius for fallback pattern matching

### Frame Extraction Parameters (`extract_frames`)

- `every_n_frames` (default: 1): Extract every Nth frame from the video
- `skip_start_frames` (default: 0): Skip N frames from the start
- `skip_start_seconds` (default: 0.0): Skip N seconds from the start
- `auto_detect_focus` (default: True): Automatically detect camera focus and skip unfocused frames
- `auto_detect_motion` (default: True): Automatically detect camera movement/shake and skip unstable frames
- `focus_threshold` (default: 50): Laplacian variance threshold for focus detection
- `motion_threshold` (default: 1000): Optical flow threshold for motion detection
- `min_consecutive_stable` (default: 15): Minimum consecutive stable frames required

### Heavy Processing Recommendations

For better marking quality, consider these parameter adjustments:

- `every_n_frames=1` (extract all frames)
- `pattern_height=15`, `pattern_width=15` (larger patterns)
- `pattern_top_grid=7`, `pattern_bottom_grid=3`, `num_divisions=75`, `min_gap=25`, `dbscan_eps=0.5`

### Quality Improvements

- **Higher temporal resolution**: All frames captured for smoother elongation curves
- **Better pattern matching**: Larger search areas and patterns improve robustness
- **More reliable detection**: Stricter filtering reduces false positives
- **Improved accuracy**: More thorough pattern selection from more frames
- **Automatic focus detection**: Skips unfocused frames automatically for better quality
- **Flexible skipping**: Both frame-based and time-based skipping options

### Troubleshooting

- **Check disk space**: Heavy processing generates many more files
- **Monitor memory usage**: Larger parameters require more RAM
- **Verify video quality**: Heavy processing is more sensitive to poor video quality
- **Check parameter compatibility**: Some parameter combinations may not work well together

### Performance Optimization

- **Use SSD storage**: Faster I/O for frame extraction
- **Increase RAM**: More memory for pattern analysis
- **Close other applications**: Free up CPU resources
- **Monitor system resources**: Use `htop` or similar tools

## Advanced Implementation Notes

### Subpixel Accuracy for Pattern Matching and Marker Detection

- The system uses subpixel refinement (parabolic fitting) to estimate marker and pattern positions with precision finer than a single pixel, improving measurement smoothness and accuracy.
- Subpixel logic is applied in both pattern matching and marker detection, using a 3-point parabola fit to the matching scores or gradient strengths.

### Marker Detection Approach and Robustness

- Marker detection is based on vertical gradient analysis (Sobel filter) and thresholding, with optional consensus/jittered detection for robustness.
- Pattern matching is used after initial detection for improved reliability.
- Suggestions for further robustness include adaptive histogram equalization, denoising, Canny edge detection, and consensus-based or ML-based marker localization.

### Fast Pattern Matching with OpenCV

- Pattern matching uses OpenCV's `cv2.matchTemplate` with the `TM_CCOEFF_NORMED` method for fast, robust, and normalized correlation-based matching.
- Subpixel accuracy is maintained by fitting a parabola to the correlation surface around the best match.
- Manual normalization/denoising is not needed, as OpenCV handles this internally.

### Multi-Reference Pattern Tracking and Aggregation

- The system can track multiple consistent pattern pairs (multi-reference), storing per-pair results and aggregating elongation values (e.g., median) for robust final measurement.
- Aggregated results are used for analysis and reporting, and per-pair results can be saved for diagnostics.

## What the System Does

- Extracts frames from video and detects stable, focused periods
- Extracts and matches natural texture patterns on the rebar
- Tracks elongation using robust clustering and median filtering
- Converts elongation to mm using user calibration
- Provides web and CLI interfaces for processing and visualization
- Generates plots, CSVs, and PDF reports for further analysis

## License

MIT License
