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

# Delete entire results folder
cleanall:
	rm -rf results/
