# Multi-Reference Pattern Tracking Plan

## 1. Pattern Pair Selection

- After filtering, keep all consistent pattern pairs (e.g., top N).

## 2. Per-Pattern-Pair Tracking

- For each consistent pattern pair:
  - Track yellow/blue lines for all frames using that pair.
  - Store the results (positions, elongation, etc.) for each frame and each pattern pair.

## 3. Data Storage

- Save a separate CSV for each pattern pair (e.g., `elongation_data_pair_0.csv`, ...).
- Each CSV contains per-frame results for that pattern pair.

## 4. Frame Visualization

- Save marked frames for each pattern pair in separate folders (e.g., `elongation_marked_pair_0/`).
- Optionally, only save the final aggregated/median result as marked frames.

## 5. Aggregation

- For each frame, aggregate elongation values across all pattern pairs (e.g., median, mean, or robust filter).
- Save the aggregated result as a new CSV (e.g., `elongation_data_aggregated.csv`).

## 6. Analysis

- Run the analysis pipeline (e.g., `analyze_elongation.py`) on the aggregated CSV for robust results.
- Optionally, compare results for each pattern pair to diagnose outliers.

## Naming/Organization Suggestions

- Output folders:
  - `elongation_marked_pair_0/`, `elongation_marked_pair_1/`, ...
- Output CSVs:
  - `elongation_data_pair_0.csv`, ..., `elongation_data_aggregated.csv`
- Aggregation:
  - Use median unless you have a reason to prefer mean or another robust statistic.

## Summary Table

| Step              | Output/Action                          |
| ----------------- | -------------------------------------- |
| Pattern selection | All consistent pairs (top N)           |
| Per-pair tracking | Save per-frame results for each pair   |
| Data storage      | CSV per pair                           |
| Aggregation       | Median/mean per frame across all pairs |
| Visualization     | Save marked frames per pair            |
| Analysis          | Run on aggregated CSV                  |
