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

def detect_gauge_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    strip = gray[:, w//2-1:w//2+1]
    brightness = np.mean(strip, axis=1)
    gradient = np.diff(brightness)

    top_section = gradient[:int(0.4 * h)]
    top_y = np.where(top_section < -10)[0]
    top_y = int(top_y[0]) if len(top_y) > 0 else 100

    bottom_section = gradient[int(0.6 * h):]
    bottom_y = np.where(bottom_section < -10)[0]
    bottom_y = int(0.6 * h) + int(bottom_y[-1]) if len(bottom_y) > 0 else h - 100

    return top_y, bottom_y

def draw_reference_grid(img, ref_top, ref_bottom, ref_distance):
    h, w = img.shape[:2]
    # Draw extended 10% grid lines from ref_top to bottom of the image
    for i in range(11):
        y = int(ref_top + i * (ref_distance / 10))
        cv2.line(img, (0, y), (w, y), color_grid, 1)

    # Extend grid above top and below bottom
    for y in range(0, ref_top, int(ref_distance / 10)):
        cv2.line(img, (0, y), (w, y), color_grid, 1)
    for y in range(ref_bottom, h, int(ref_distance / 10)):
        cv2.line(img, (0, y), (w, y), color_grid, 1)


def process_images():
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted(f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png")))
    data = []

    first_img = cv2.imread(os.path.join(input_folder, frames[0]))
    ref_top, ref_bottom = detect_gauge_markers(first_img)
    ref_distance = ref_bottom - ref_top

    for frame in frames:
        img_path = os.path.join(input_folder, frame)
        img = cv2.imread(img_path)
        top_y, bottom_y = detect_gauge_markers(img)
        curr_distance = bottom_y - top_y
        elongation_percent = (curr_distance / ref_distance) * 100

        # Draw %10 interval grid
        draw_reference_grid(img, ref_top, ref_bottom, ref_distance)

      # Yellow base gauge
        cv2.line(img, (0, ref_top), (img.shape[1], ref_top), color_ref, 2)
        cv2.line(img, (0, ref_bottom), (img.shape[1], ref_bottom), color_ref, 2)
        cv2.putText(img, f"{ref_distance}px (100%)", (10, ref_top - 10),
                    font, font_scale, color_ref, 2)

        # Blue current gauge
        cv2.line(img, (0, top_y), (img.shape[1], top_y), color_curr, 2)
        cv2.line(img, (0, bottom_y), (img.shape[1], bottom_y), color_curr, 2)
        cv2.putText(img, f"{curr_distance}px ({elongation_percent:.1f}%)",
                    (img.shape[1] - 260, top_y - 10), font, font_scale, color_curr, 2)


        # Save image
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
