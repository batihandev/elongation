import os
import cv2
import numpy as np
import pandas as pd

# Parameters
input_folder = "output_frames"
output_folder = "elongation_marked_frames"
csv_output_path = "elongation_data.csv"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
color_ref = (0, 255, 255)    # Yellow
color_curr = (255, 100, 100) # Blue
color_grid = (200, 200, 200)
color_middle = (0, 0, 255)   # Red

transition_threshold = 25


def detect_rebar_center_x(gray):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    energy = np.sum(np.abs(sobel), axis=0)

    # Detect left and right edges of high vertical energy band
    w = gray.shape[1]
    middle_band = energy[w//4:w*3//4]

    threshold = 0.4 * np.max(middle_band)
    indices = np.where(middle_band > threshold)[0]
    
    if len(indices) == 0:
        return w // 2  # fallback

    left = indices[0] + w//4
    right = indices[-1] + w//4
    return (left + right) // 2


def detect_markers_5x5(gray, center_x):
    h, w = gray.shape

    def scan(direction='top'):
        # Safer range: 2 to h-3 allows for y-2:y+3 (5x5 patch)
        range_y = range(2, h - 2) if direction == 'top' else range(h - 3, 1, -1)
        for y in range_y:
            patch = gray[y - 2:y + 3, center_x - 2:center_x + 3]
            if patch.shape != (5, 5):  # Skip edge errors
                continue
            diffs = np.abs(patch - np.median(patch))
            changed = np.sum(diffs > transition_threshold)
            if changed >= 13:  # More than half of 25 pixels
                return y
        return None

    return scan('top'), scan('bottom')



def draw_reference_grid(img, ref_top, ref_bottom, ref_distance):
    h, w = img.shape[:2]
    for i in range(11):
        y = int(ref_top + i * (ref_distance / 10))
        cv2.line(img, (0, y), (w, y), color_grid, 1)
    for y in range(0, ref_top, int(ref_distance / 10)):
        cv2.line(img, (0, y), (w, y), color_grid, 1)
    for y in range(ref_bottom, h, int(ref_distance / 10)):
        cv2.line(img, (0, y), (w, y), color_grid, 1)

def process_images():
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png")))
    data = []
    ref_top = None
    ref_bottom = None
    ref_distance = None

    for i, frame in enumerate(frames):
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center_x = detect_rebar_center_x(gray)
        top_y, bottom_y = detect_markers_5x5(gray, center_x)
        print(f"BLUE MARKERS: top={top_y}, bottom={bottom_y}, distance={bottom_y - top_y}")

        if top_y is None or bottom_y is None:
            print(f"⚠ Skipping frame {frame}: markers not found")
            continue
        if ref_top is None and top_y is not None and bottom_y is not None and (bottom_y - top_y) > 80:
            ref_top = top_y
            ref_bottom = bottom_y
            ref_distance = ref_bottom - ref_top
            print(f"✅ Reference markers set at frame {i}: top={ref_top}, bottom={ref_bottom}, distance={ref_distance}px")


        curr_distance = bottom_y - top_y
        elongation_percent = (curr_distance / ref_distance) * 100

        # Draw center red line
        cv2.line(img, (center_x, 0), (center_x, img.shape[0]), color_middle, 1)

        # Draw reference grid
        draw_reference_grid(img, ref_top, ref_bottom, ref_distance)

        # Draw reference (yellow) gauge lines
        cv2.line(img, (0, ref_top), (img.shape[1], ref_top), color_ref, 2)
        cv2.line(img, (0, ref_bottom), (img.shape[1], ref_bottom), color_ref, 2)
        cv2.putText(img, f"{ref_distance}px (100%)", (10, ref_top - 10), font, font_scale, color_ref, 2)

        # Draw current (blue) gauge lines
        cv2.line(img, (0, top_y), (img.shape[1], top_y), color_curr, 2)
        cv2.line(img, (0, bottom_y), (img.shape[1], bottom_y), color_curr, 2)
        cv2.putText(img, f"{curr_distance}px ({elongation_percent:.1f}%)", (img.shape[1] - 260, top_y - 10),
                    font, font_scale, color_curr, 2)
        # === Draw visual pixel scale bars ===
        bar_y = img.shape[0] - 20  # Bottom margin
        bar_margin_right = 20
        font_thickness = 1
        bar_thickness = 2

        # Draw 5px bar
        bar5_len = 5
        bar5_x_start = img.shape[1] - bar5_len - bar_margin_right-45
        cv2.line(img, (bar5_x_start, bar_y), (bar5_x_start + bar5_len, bar_y), (255, 255, 255), bar_thickness)
        cv2.putText(img, "5 px", (bar5_x_start - 5, bar_y - 10), font, font_scale, (255, 255, 255), font_thickness)

        # Draw 50px bar
        bar50_len = 50
        bar50_x_start = img.shape[1] - bar50_len - bar_margin_right
        cv2.line(img, (bar50_x_start, bar_y - 30), (bar50_x_start + bar50_len, bar_y - 30), (255, 255, 255), bar_thickness)
        cv2.putText(img, "50 px", (bar50_x_start - 5, bar_y - 40), font, font_scale, (255, 255, 255), font_thickness)


        # Save marked image
        output_path = os.path.join(output_folder, frame)
        cv2.imwrite(output_path, img)

        # Save data
        timestamp = frame.split('_')[-1].replace('s.jpg', '').replace('s.png', '')
        data.append({
            "frame": frame,
            "timestamp_s": float(timestamp),
            "ref_top_px": ref_top,
            "ref_bottom_px": ref_bottom,
            "curr_top_px": top_y,
            "curr_bottom_px": bottom_y,
            "elongation_px": curr_distance - ref_distance,
            "elongation_percent": elongation_percent
        })

    pd.DataFrame(data).to_csv(csv_output_path, index=False)
    print(f"✔ All done! Marked frames in '{output_folder}', data in '{csv_output_path}'")

# Run
if __name__ == "__main__":
    process_images()
