.PHONY: install run clean mark elongation_data analyze draw_clamps perfect batch runserver testcurl sample1 sample2 sample1_heavy sample2_heavy sample2_sensitive

install:
	pip install -r requirements.txt

run:
	python backend/main.py

clean:
	rm -rf results/output_frames_* results/elongation_marked_* results/*.csv results/*.png uploads/

mark:
	python elongation/mark_frames.py

elongation_data:
	python elongation/clean_elongation_data.py

analyze:
	python elongation/analyze_elongation.py

draw_clamps:
	python elongation/draw_clamps.py

perfect:
	python elongation/almost_perfect.py

batch:
	python elongation/batch_runner.py

runserver:
	uvicorn backend.main:app --reload

testcurl:
	curl -X POST http://localhost:8000/process/ \
	-F "video=@sample/40kn-2.mp4" \
	-F "every_n_frames=5" \
	-F "min_elong=100" \
	-F "max_elong=140"

# Simple commands to process sample videos
sample1:
	python process_video.py 1

sample2:
	python process_video.py 2

# Sensitive motion detection for sample2 (detects subtle movements)
sample2_sensitive:
	python process_video.py 2_sensitive

# Heavy processing targets with optimized parameters for better marking quality
# These use every_n_frames=1 and enhanced pattern detection parameters
sample1_heavy:
	python -c "from elongation.extract_frames import extract_frames; from elongation.mark_frames import process_images; from elongation.analyze_elongation import process_and_plot; import pandas as pd; import os; \
	video_path='sample/40kn-1.mp4'; base_name='40kn-1'; \
	frames_dir=f'results/output_frames_{base_name}'; marked_dir=f'results/elongation_marked_{base_name}'; \
	csv_path=f'results/elongation_data_{base_name}.csv'; final_csv=f'results/elongation_final_{base_name}.csv'; \
	plot_path=f'results/elongation_plot_{base_name}.png'; \
	print('🎬 Heavy processing sample1 with every_n_frames=1, focus/motion detection, and enhanced parameters...'); \
	extraction_result = extract_frames(video_path, frames_dir, every_n_frames=1, \
		auto_detect_focus=True, auto_detect_motion=True, focus_threshold=50, motion_threshold=200, \
		edge_threshold=30, bg_threshold=20, min_consecutive_stable=10, skip_start_seconds=2.0); \
	print(f'   Frames saved: {extraction_result[\"frames_saved\"]}, Skipped: {extraction_result[\"frames_skipped\"]}'); \
	process_images(frames_dir, marked_dir, csv_path, \
		max_pattern_height=15, pattern_width=10, search_margin_y=8, search_margin_x=8, \
		pattern_capture_frames=15, pattern_capture_step=3, top_n_to_keep=8, \
		prune_threshold=5e6, scan_width=12, threshold_ratio=0.25, \
		min_valid_distance=60, max_band_thickness=12); \
	df = pd.read_csv(csv_path); \
	result = process_and_plot(df, final_csv, plot_path, min_elongation=100, max_elongation=140); \
	print('✅ Heavy processing complete!')"

sample2_heavy:
	python -c "from elongation.extract_frames import extract_frames; from elongation.mark_frames import process_images; from elongation.analyze_elongation import process_and_plot; import pandas as pd; import os; \
	video_path='sample/40kn-2.mp4'; base_name='40kn-2'; \
	frames_dir=f'results/output_frames_{base_name}'; marked_dir=f'results/elongation_marked_{base_name}'; \
	csv_path=f'results/elongation_data_{base_name}.csv'; final_csv=f'results/elongation_final_{base_name}.csv'; \
	plot_path=f'results/elongation_plot_{base_name}.png'; \
	print('🎬 Heavy processing sample2 with every_n_frames=1, focus/motion detection, and enhanced parameters...'); \
	extraction_result = extract_frames(video_path, frames_dir, every_n_frames=1, \
		auto_detect_focus=True, auto_detect_motion=True, focus_threshold=50, motion_threshold=200, \
		edge_threshold=30, bg_threshold=20, min_consecutive_stable=10, skip_start_seconds=2.0); \
	print(f'   Frames saved: {extraction_result[\"frames_saved\"]}, Skipped: {extraction_result[\"frames_skipped\"]}'); \
	process_images(frames_dir, marked_dir, csv_path, \
		max_pattern_height=15, pattern_width=10, search_margin_y=8, search_margin_x=8, \
		pattern_capture_frames=15, pattern_capture_step=3, top_n_to_keep=8, \
		prune_threshold=5e6, scan_width=12, threshold_ratio=0.25, \
		min_valid_distance=60, max_band_thickness=12); \
	df = pd.read_csv(csv_path); \
	result = process_and_plot(df, final_csv, plot_path, min_elongation=100, max_elongation=140); \
	print('✅ Heavy processing complete!')"

# Delete entire results folder
cleanall:
	rm -rf results/
