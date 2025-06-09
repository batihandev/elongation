from typing import Optional
from pydantic import BaseModel

class ElongationOptions(BaseModel):
    skip_start: Optional[int] = 0
    skip_end: Optional[int] = 0
    yield_strength: Optional[float] = 420   # MPa
    rebar_diameter: Optional[int] = 16      # mm
    font_scale: Optional[float] = 0.6
    pattern_width: Optional[int] = 8
    max_pattern_height: Optional[int] = 12
    search_margin_y: Optional[int] = 5
    search_margin_x: Optional[int] = 5
    pattern_capture_frames: Optional[int] = 10
    pattern_capture_step: Optional[int] = 5
    prune_threshold: Optional[float] = 1e7
    top_n_to_keep: Optional[int] = 5
    scan_width: Optional[int] = 10
    threshold_ratio: Optional[float] = 0.3
    min_valid_distance: Optional[int] = 50
    max_band_thickness: Optional[int] = 10
