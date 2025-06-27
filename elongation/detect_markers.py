import cv2
import numpy as np
from elongation.subpixel import subpixel_refine

def detect_markers(gray, center_x, scan_width=10, threshold_ratio=0.3, min_valid_distance=50, max_band_thickness=10):
    h, w = gray.shape
    x1, x2 = max(center_x - scan_width, 0), min(center_x + scan_width, w)

    strip = gray[:, x1:x2]
    strip = cv2.equalizeHist(strip)

    sobel_vertical = cv2.Sobel(strip, cv2.CV_64F, 0, 1, ksize=3)
    gradient_strength = np.abs(sobel_vertical).sum(axis=1)
    threshold = np.max(gradient_strength) * threshold_ratio

    def find_band(start_y, end_y, step):
        for y in range(start_y, end_y, step):
            if gradient_strength[y] > threshold:
                for dy in range(1, max_band_thickness):
                    next_y = y + dy * step
                    if 0 <= next_y < h and gradient_strength[next_y] > threshold:
                        # Subpixel refinement for the band position
                        band_y = (y + next_y) // 2
                        band_y_subpixel = subpixel_refine(gradient_strength, band_y, mode='array')
                        return band_y_subpixel
        return None

    # Focus on middle section: 30% to 70% of rebar height
    # This corresponds to around the 40% and 60% grid lines
    mid_start = int(h * 0.3)  # 30% from top
    mid_end = int(h * 0.7)    # 70% from top
    mid_center = (mid_start + mid_end) // 2
    
    # Search for top marker in upper middle section (30% to 50%)
    top_y = find_band(mid_start, mid_center, 1)
    
    # Search for bottom marker in lower middle section (50% to 70%)
    bottom_y = find_band(mid_end, mid_center, -1)

    if top_y is not None and bottom_y is not None:
        if (bottom_y - top_y) < min_valid_distance:
            return None, None

    return top_y, bottom_y
