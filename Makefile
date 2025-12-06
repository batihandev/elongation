.PHONY: install run clean mark elongation_data analyze draw_clamps perfect batch runserver testcurl sample1 sample2 cleanall

# Install dependencies
install:
	python -m pip install -r requirements.txt

# Run backend main
run:
	python -m elongation_rebar.backend.main

# Clean generated data
clean:
	rm -rf results/output_frames_* results/elongation_marked_* results/*.csv results/*.png uploads/

# Mark frames
mark:
	python -m elongation_rebar.elongation.mark_frames

# Clean elongation data
elongation_data:
	python -m elongation_rebar.elongation.clean_elongation_data

# Analyze elongation data
analyze:
	python -m elongation_rebar.elongation.analyze_elongation

# Draw clamps
draw_clamps:
	python -m elongation_rebar.elongation.draw_clamps

# Almost-perfect mode
perfect:
	python -m elongation_rebar.elongation.almost_perfect

# Batch runner
batch:
	python -m elongation_rebar.elongation.batch_runner

# Run FastAPI server
runserver:
	uvicorn elongation_rebar.backend.main:app --reload

# Test curl upload
testcurl:
	curl -X POST http://localhost:8000/process/ \
		-F "video=@sample/40kn-2.mp4" \
		-F "every_n_frames=5" \
		-F "min_elong=100" \
		-F "max_elong=140"

# Sample video processors
sample1:
	python process_video.py 1

sample2:
	python process_video.py 2

# Wipe everything
cleanall:
	rm -rf results/

activate:
	source .venv/bin/activate