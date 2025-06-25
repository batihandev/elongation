from fastapi import FastAPI, UploadFile, File, Form,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil, os, pandas as pd
from elongation.extract_frames import extract_frames
from elongation.mark_frames import process_images
from elongation.analyze_elongation import process_and_plot
import concurrent.futures
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
from fastapi.staticfiles import StaticFiles
import threading
from elongation.pixel_to_mm import  create_pixel_to_mm_csv, plot_percent_with_mm_axis
import re

processing_cancelled = threading.Event()
RESULTS_DIR = "results"

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
clients: List[WebSocket] = []
async def broadcast_progress(msg: str):
    for client in clients:
        try:
            await client.send_text(msg)
        except:
            pass
def make_progress_callback(loop):
    def cb(p, message):
        text = f"{int(p * 100)}% - {message}"
        # Schedule on the main event loop, even from other threads
        loop.call_soon_threadsafe(asyncio.create_task, broadcast_progress(text))
    return cb

def normalize_video_name(name: str) -> str:
    """
    Normalize a video name:
    - Replace %20 and spaces with underscores
    - Strip known video file extensions
    - Preserve internal dots (e.g., timestamps)
    """
    name = name.strip().replace("%20", "_").replace(" ", "_")
    # Remove extension only if it matches known video formats
    return re.sub(r"\.(mp4|mov|avi|mkv)$", "", name, flags=re.IGNORECASE)



app = FastAPI()
@app.post("/process/")
async def process_video(
    video: UploadFile = File(...),
    every_n_frames: int = Form(5),
    min_elong: float = Form(100),
    max_elong: float = Form(140),
    skip_start: int = Form(0),
    skip_end: int = Form(0),
    skip_start_seconds: float = Form(0.0),
    auto_detect_focus: bool = Form(True),
    auto_detect_motion: bool = Form(True),
    focus_threshold: float = Form(50.0),
    motion_threshold: float = Form(200.0),
    min_consecutive_stable: int = Form(10),
    font_scale: float = Form(0.6),
    pattern_width: int = Form(8),
    max_pattern_height: int = Form(12),
    search_margin_y: int = Form(5),
    search_margin_x: int = Form(5),
    pattern_capture_frames: int = Form(10),
    pattern_capture_step: int = Form(5),
    prune_threshold: float = Form(1e7),
    top_n_to_keep: int = Form(5),
    scan_width: int = Form(10),
    threshold_ratio: float = Form(0.3),
    min_valid_distance: int = Form(50),
    max_band_thickness: int = Form(10)

):
    loop = asyncio.get_running_loop()
    callback = make_progress_callback(loop)
    safe_base_name = normalize_video_name(video.filename or "unknown_video")  # Normalize video name
    video_path = f"uploads/{video.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    frames_dir = f"results/output_frames_{safe_base_name}"
    marked_dir = f"results/elongation_marked_{safe_base_name}"
    csv_path = f"results/elongation_data_{safe_base_name}.csv"
    final_csv_path = f"results/elongation_final_{safe_base_name}.csv"
    plot_path = f"results/elongation_plot_{safe_base_name}.png"
    def blocking_processing():
        if processing_cancelled.is_set():
            raise RuntimeError("Processing cancelled before start.")
        processing_cancelled.clear()
        
        # Extract frames with focus detection and flexible skipping
        extraction_result = extract_frames(
            video_path, 
            frames_dir, 
            every_n_frames=every_n_frames, 
            skip_start_frames=skip_start,
            skip_start_seconds=skip_start_seconds,
            auto_detect_focus=auto_detect_focus,
            auto_detect_motion=auto_detect_motion,
            focus_threshold=int(focus_threshold),
            motion_threshold=int(motion_threshold),
            edge_threshold=30,  # Match test parameters
            bg_threshold=20,    # Match test parameters
            min_consecutive_stable=min_consecutive_stable,
            progress_callback=callback,
            cancel_event=processing_cancelled
        )
        
        process_images(
            input_folder=frames_dir,
            output_folder=marked_dir,
            csv_output_path=csv_path,
            skip_start_frames=0,  # Already handled in extract_frames
            skip_end_frames=skip_end,
            font_scale=font_scale,
            pattern_width=pattern_width,
            max_pattern_height=max_pattern_height,
            search_margin_y=search_margin_y,
            search_margin_x=search_margin_x,
            pattern_capture_frames=pattern_capture_frames,
            pattern_capture_step=pattern_capture_step,
            prune_threshold=prune_threshold,
            top_n_to_keep=top_n_to_keep,
            scan_width=scan_width,
            threshold_ratio=threshold_ratio,
            min_valid_distance=min_valid_distance,
            max_band_thickness=max_band_thickness,
            progress_callback=callback,
            cancel_event=processing_cancelled
        )
        if not os.path.exists(csv_path):
            raise RuntimeError("No data generated — check marker detection.")
        df = pd.read_csv(csv_path)
        result = process_and_plot(
            df,
            output_csv=final_csv_path,
            plot_path=plot_path,
            min_elongation=int(min_elong),
            max_elongation=int(max_elong),
            progress_callback=callback,
            cancel_event=processing_cancelled
        )
        if result is None:
            result_df, yield_time, yield_elongation = None, None, None
        else:
            result_df, yield_time, yield_elongation = result
        return csv_path, final_csv_path, plot_path, yield_time, yield_elongation, extraction_result

    csv_path, final_csv_path, plot_path, yield_time, yield_elongation, extraction_result = await loop.run_in_executor(executor, blocking_processing)

    return {
        "status": "Processed",
        "csv_data": csv_path,
        "final_data": final_csv_path,
        "plot": plot_path,
        "yield_time_s": yield_time,
        "yield_elongation_percent": yield_elongation,
        "extraction_info": {
            "frames_saved": extraction_result.get('frames_saved', 0),
            "frames_skipped": extraction_result.get('frames_skipped', 0),
            "time_skipped": extraction_result.get('time_skipped', 0.0),
            "focus_detected": extraction_result.get('focus_info') is not None
        }
    }
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keeps connection alive
    except WebSocketDisconnect:
        clients.remove(websocket)
        
os.makedirs("results", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")

@app.post("/stop/")
async def stop_processing():
    processing_cancelled.set()
    return {"status": "stopping"}

from fastapi import Query
from fastapi.responses import FileResponse

@app.get("/first_marked_image/")
def get_first_marked_image(video_name: str = Query(...)):
    safe_base_name = normalize_video_name(video_name)  # Normalize video name
    folder = f"results/elongation_marked_{safe_base_name}"
    files = sorted(os.listdir(folder))
    if not files:
        return {"error": "No marked images found"}
    first_img_path = os.path.join(folder, files[0])
    return FileResponse(first_img_path)

@app.post("/pixel_to_mm/")
async def pixel_to_mm_endpoint(request: Request):
    try:
        data = await request.json()
        video_name = data.get("videoName")
        x1 = data.get("x1")
        y1 = data.get("y1")
        x2 = data.get("x2")
        y2 = data.get("y2")
        p1 = {"x": x1, "y": y1}
        p2 = {"x": x2, "y": y2}
        real_distance_mm = float(data.get("mmValue"))

        if not all([video_name, p1, p2, real_distance_mm]):
            return JSONResponse(status_code=400, content={"error": "Missing required fields."})

        base_name = normalize_video_name(video_name)

        data_csv_path = f"results/elongation_final_{base_name}.csv"
        pixel_to_mm_csv_path = f"results/pixel_to_mm_{base_name}.csv"
        output_plot_path = f"results/elongation_plot_mm_{base_name}.png"

        create_pixel_to_mm_csv(base_name, p1, p2, real_distance_mm)
        plot_percent_with_mm_axis(data_csv_path, pixel_to_mm_csv_path, output_plot_path)

        return {
            "status": "success",
            "plot": output_plot_path,
            "pixel_to_mm_csv": pixel_to_mm_csv_path,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/list_processed/")
async def list_processed_videos():
    try:
        files = os.listdir(RESULTS_DIR)
        # Filter unique base names by detecting the elongation_data CSV files
        base_names = set()
        for f in files:
            if f.startswith("elongation_data_") and f.endswith(".csv"):
                base_names.add(f[len("elongation_data_"):-len(".csv")])
        return {"videos": sorted(base_names)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/delete_processed/")
async def delete_processed_video(video_name: str):
    try:
        base_name = video_name
        files_to_delete = [
            f"elongation_data_{base_name}.csv",
            f"elongation_final_{base_name}.csv",
            f"elongation_plot_{base_name}.png",
            f"elongation_plot_mm_{base_name}.png",
            f"pixel_to_mm_{base_name}.csv",
        ]
        # Delete CSV, PNG, pixel_to_mm files
        for fname in files_to_delete:
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                os.remove(path)
        # Delete folder with marked images
        folder = os.path.join(RESULTS_DIR, f"elongation_marked_{base_name}")
        if os.path.isdir(folder):
            import shutil
            shutil.rmtree(folder)
        # Delete folder with output frames
        folder2 = os.path.join(RESULTS_DIR, f"output_frames_{base_name}")
        if os.path.isdir(folder2):
            import shutil
            shutil.rmtree(folder2)
        return {"status": "deleted", "video_name": video_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})