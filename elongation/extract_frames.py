import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_folder, every_n_frames=1, progress_callback=None,cancel_event=None):
    def notify_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)
        else:
            print(f"{int(p*100)}% - {msg}")

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("FPS is 0, cannot proceed.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    saved_index = 0
    notify_progress(0.0, f"Opened video: {video_path}, FPS: {fps}")

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_index % every_n_frames == 0:
                timestamp = frame_index / fps
                filename = os.path.join(output_folder, f"frame_{saved_index:04d}_{timestamp:.2f}s.jpg")
                cv2.imwrite(filename, frame)
                saved_index += 1
            if cancel_event and cancel_event.is_set():
                notify_progress(1.0, "Processing cancelled by user.")
                break

            frame_index += 1
            pbar.update(1)

    cap.release()
    notify_progress(0.05, f"Total frames saved: {saved_index}")

if __name__ == "__main__":
    extract_frames("40kn-2.mp4", "output_frames", every_n_frames=5)