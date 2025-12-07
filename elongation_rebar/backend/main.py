import os
import re
import shutil
import threading
import asyncio
import hashlib
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
from typing import List, Dict
import io
from fastapi.responses import StreamingResponse

import pandas as pd
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Query,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..elongation.extract_frames import extract_frames
from ..elongation.mark_frames import process_images
from ..elongation.analyze_elongation import process_and_plot
from ..elongation.pixel_to_mm import (
    create_pixel_to_mm_csv,
    plot_percent_with_mm_axis,
)

# -------------------------------------------------------------------
# Globals & config
# -------------------------------------------------------------------

processing_cancelled = threading.Event()

RESULTS_DIR = "results"

# Simple IP-based rate limit: 2 videos / 24h
RATE_LIMIT_MAX = 100
RATE_LIMIT_WINDOW = timedelta(hours=24)
_rate_limit_state: Dict[str, Dict] = {}

# Only one heavy processing job at a time
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

clients: List[WebSocket] = []


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def cleanup_results_for_base(base_name: str) -> None:
    """
    Remove all result files/folders for a given base_name.
    base_name is the normalized original filename (without extension).
    """
    files_to_delete = [
        f"elongation_data_{base_name}.csv",
        f"elongation_final_{base_name}.csv",
        f"elongation_plot_{base_name}.png",
        f"elongation_plot_mm_{base_name}.png",
        f"pixel_to_mm_{base_name}.csv",
    ]

    for fname in files_to_delete:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            os.remove(path)

    dirs_to_delete = [
        os.path.join(RESULTS_DIR, f"elongation_marked_{base_name}"),
        os.path.join(RESULTS_DIR, f"output_frames_{base_name}"),
    ]

    for d in dirs_to_delete:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = datetime.utcnow()

    entry = _rate_limit_state.get(ip)
    if not entry or entry["reset_at"] <= now:
        entry = {"count": 0, "reset_at": now + RATE_LIMIT_WINDOW}
        _rate_limit_state[ip] = entry

    if entry["count"] >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX} videos per 24 hours.",
        )

    entry["count"] += 1


async def broadcast_progress(msg: str) -> None:
    for client in clients:
        try:
            await client.send_text(msg)
        except Exception:
            # ignore dead sockets
            pass


def make_progress_callback(loop: asyncio.AbstractEventLoop):
    def cb(p: float, message: str) -> None:
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
    adjusted_name = re.sub(r"\.(mp4|mov|avi|mkv)$", "", name, flags=re.IGNORECASE)
    # create deterministic string from original name to avoid collisions
    adjusted_name = hashlib.md5(name.encode()).hexdigest()[:8]
    print(f"Normalized video name: {adjusted_name}")  # <-- log here
    return adjusted_name


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI()


@app.post("/process")
async def process_video(
    request: Request,
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
    pattern_width: int = Form(15),
    pattern_height: int = Form(15),
    pattern_top_grid: int = Form(7),
    pattern_bottom_grid: int = Form(3),
    num_divisions: int = Form(75),
    min_gap: int = Form(25),
    dbscan_eps: float = Form(1.0),
):
    # --- rate limit per IP ---
    check_rate_limit(request)

    # --- sanitize original filename (used as logical "video name") ---
    original_name = Path(video.filename or "upload.mp4").name
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv"}
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Unsupported video type.")

    # base_name = normalized original filename without extension
    base_name = normalize_video_name(os.path.splitext(original_name)[0])

    # --- store uploaded file with random name (don't trust original) ---
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", base_name + ext)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    loop = asyncio.get_running_loop()
    callback = make_progress_callback(loop)
    frames_dir = os.path.join(RESULTS_DIR, f"output_frames_{base_name}")
    marked_dir = os.path.join(RESULTS_DIR, f"elongation_marked_{base_name}")
    csv_path = os.path.join(RESULTS_DIR, f"elongation_data_{base_name}.csv")
    final_csv_path = os.path.join(RESULTS_DIR, f"elongation_final_{base_name}.csv")
    plot_path = os.path.join(RESULTS_DIR, f"elongation_plot_{base_name}.png")

    def blocking_processing():
        if processing_cancelled.is_set():
            raise RuntimeError("Processing cancelled before start.")
        processing_cancelled.clear()

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
            edge_threshold=30,
            bg_threshold=20,
            min_consecutive_stable=min_consecutive_stable,
            progress_callback=callback,
            cancel_event=processing_cancelled,
        )

        process_images(
            input_folder=frames_dir,
            output_folder=marked_dir,
            csv_output_path=csv_path,
            skip_start_frames=0,  # Already handled in extract_frames
            skip_end_frames=skip_end,
            font_scale=font_scale,
            pattern_width=pattern_width,
            pattern_height=pattern_height,
            pattern_top_grid=pattern_top_grid,
            pattern_bottom_grid=pattern_bottom_grid,
            num_divisions=num_divisions,
            min_gap=min_gap,
            dbscan_eps=float(dbscan_eps),
            progress_callback=callback,
            cancel_event=processing_cancelled,
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
            cancel_event=processing_cancelled,
        )

        if result is None:
            result_df, yield_time, yield_elongation = None, None, None
        else:
            result_df, yield_time, yield_elongation = result

        return csv_path, final_csv_path, plot_path, yield_time, yield_elongation, extraction_result

    try:
        (
            csv_path,
            final_csv_path,
            plot_path,
            yield_time,
            yield_elongation,
            extraction_result,
        ) = await loop.run_in_executor(executor, blocking_processing)

    except Exception as e:
        # Any failure: wipe all results for this base_name and return a 400
        cleanup_results_for_base(base_name)
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}")

    finally:
        # Always delete uploaded video file to save disk
        if os.path.exists(video_path):
            os.remove(video_path)

    return {
        "status": "Processed",
        "csv_data": csv_path,
        "final_data": final_csv_path,
        "plot": plot_path,
        "yield_time_s": yield_time,
        "yield_elongation_percent": yield_elongation,
        "extraction_info": {
            "frames_saved": extraction_result.get("frames_saved", 0),
            "frames_skipped": extraction_result.get("frames_skipped", 0),
            "time_skipped": extraction_result.get("time_skipped", 0.0),
            "focus_detected": extraction_result.get("focus_info") is not None,
        },
        "base_name": base_name,
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


os.makedirs(RESULTS_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.post("/stop")
async def stop_processing():
    processing_cancelled.set()
    return {"status": "stopping"}


@app.get("/first_marked_image")
def get_first_marked_image(base_name: str = Query(...)):

    folder = os.path.join(RESULTS_DIR, f"elongation_marked_{base_name}")
    print(f"Looking for marked images in folder: {folder}")  # <-- log here
    if not os.path.isdir(folder):
        return JSONResponse(
            status_code=404,
            content={"error": f"No marked images folder found for '{base_name}'."},
        )

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not files:
        return JSONResponse(
            status_code=404,
            content={"error": "No marked images found in folder."},
        )

    first_img_path = os.path.join(folder, files[0])
    return FileResponse(first_img_path)


@app.post("/pixel_to_mm")
async def pixel_to_mm_endpoint(request: Request):
    try:
        data = await request.json()
        video_name = data.get("videoName")
        x1 = data.get("x1")
        y1 = data.get("y1")
        x2 = data.get("x2")
        y2 = data.get("y2")
        real_distance_mm_raw = data.get("mmValue")

        if video_name is None or real_distance_mm_raw is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing videoName or mmValue."},
            )

        base_name = video_name

        p1 = {"x": x1, "y": y1}
        p2 = {"x": x2, "y": y2}
        real_distance_mm = float(real_distance_mm_raw)

        data_csv_path = os.path.join(RESULTS_DIR, f"elongation_final_{base_name}.csv")
        pixel_to_mm_csv_path = os.path.join(RESULTS_DIR, f"pixel_to_mm_{base_name}.csv")
        output_plot_path = os.path.join(
            RESULTS_DIR, f"elongation_plot_mm_{base_name}.png"
        )

        create_pixel_to_mm_csv(base_name, p1, p2, real_distance_mm)
        plot_percent_with_mm_axis(
            data_csv_path, pixel_to_mm_csv_path, output_plot_path
        )

        return {
            "status": "success",
            "plot": output_plot_path,
            "pixel_to_mm_csv": pixel_to_mm_csv_path,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/list_processed")
async def list_processed_videos():
    try:
        files = os.listdir(RESULTS_DIR)
        # detect base_names via elongation_data CSV files
        base_names = set()
        for f in files:
            if f.startswith("elongation_data_") and f.endswith(".csv"):
                base_names.add(f[len("elongation_data_") : -len(".csv")])
        return {"videos": sorted(base_names)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/delete_processed")
async def delete_processed_video(video_name: str):
    try:
        # accept either original name or base_name
        cleanup_results_for_base(video_name)
        return {"status": "deleted", "video_name": video_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/final_data_simplified")
def get_final_data_simplified(
    base_name: str = Query(...),
    as_csv: bool = Query(False),
):
    """
    Simplified data = for each integer second, pick the row whose timestamp_s
    is closest to that second. No value modifications, just filtering.

    Uses elongation_final_{base}.csv and includes *_mm if present.
    """
    final_csv_path = os.path.join(RESULTS_DIR, f"elongation_final_{base_name}.csv")

    if not os.path.exists(final_csv_path):
        raise HTTPException(status_code=404, detail="Final data not found.")

    df = pd.read_csv(final_csv_path)

    # Ensure we have timestamp_s; if not, derive it from frame names.
    if "timestamp_s" not in df.columns:
        if "frame" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Neither 'timestamp_s' nor 'frame' column found in final CSV.",
            )

        def extract_ts(name: str):
            m = re.search(r"_(\d+(?:\.\d+)?)s", str(name))
            return float(m.group(1)) if m else None

        df["timestamp_s"] = df["frame"].map(extract_ts)
        df = df.dropna(subset=["timestamp_s"])

    # Sort by time
    df["timestamp_s"] = df["timestamp_s"].astype(float)
    df = df.sort_values("timestamp_s").reset_index(drop=True)

    # Bucket by nearest integer second, but keep original timestamp_s
    df["second"] = df["timestamp_s"].round().astype(int)
    df["dist"] = (df["timestamp_s"] - df["second"]).abs()

    # For each integer second, take row with minimum distance
    idx = df.groupby("second")["dist"].idxmin()
    simplified = df.loc[idx].sort_values("second").reset_index(drop=True)

    # Which columns to return
    has_mm = "elongation_monotonic_mm" in simplified.columns
    cols = ["timestamp_s", "elongation_monotonic"]
    if has_mm:
        cols.append("elongation_monotonic_mm")

    simplified = simplified[cols]

    if as_csv:
        buf = io.StringIO()
        simplified.to_csv(buf, index=False)
        buf.seek(0)
        filename = f"elongation_simplified_{base_name}.csv"
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return {
        "base_name": base_name,
        "rows": simplified.to_dict(orient="records"),
        "has_mm": has_mm,
    }
