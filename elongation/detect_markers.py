import cv2
import numpy as np

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
                        return (y + next_y) // 2
        return None

    mid = h // 2
    top_y = find_band(10, mid, 1)
    bottom_y = find_band(h - 10, mid, -1)

    if top_y is not None and bottom_y is not None:
        if (bottom_y - top_y) < min_valid_distance:
            return None, None

    return top_y, bottom_y
