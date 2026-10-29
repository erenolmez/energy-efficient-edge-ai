# Progress report, 29 October 2026

`main.tex` is the editable four-page LaTeX report; `main.pdf` is the compiled
presentation. The 13 and 20 October reports are preserved unchanged.

Reading guide: Study A compares compact router architectures with six resident
classifiers and eager execution. Study B implements runtime changes in full
TinyCNN routing runs (81 sweeps); its five-variant table is separate from the
isolated GPU-selection/conditional/engine-profile probes. The batch1 plot and
payback table use Study A, not an all-optimizations-combined router. Study B's
seed42 resident-Graph payback (28,621 images vs graph eligible static) is stated
separately from Study A's 2,399 images vs R101 FP32. No conclusion section is
included, as requested.

The report addresses the meeting requests: batch statistics including batch1,
scientific plots, GPU router-fitting energy and accuracy-gated training-energy
payback. It includes larger-batch results, the compact batch1 comparison,
switching/residency/dispatch investigations and training/calibration/grouping
screens. Historical 213-choice/capacity2 and new six-resident-model protocols
are explicitly distinguished. The cancelled full batch1 matrix is not claimed
complete. No inference, retraining or automation is started by these scripts.

Rebuild assets with `build_report_assets.py`, then normalize generated prose
using `format_report.py`. Compile `main.tex` with Tectonic or a compatible TeX
distribution. Required images and generated TeX tables are supplied so PDF
compilation alone does not need datasets, engines or measurement artifacts.
`report_data.json` freezes report inputs and their SHA256 hashes. Historical
five-repeat costs use medians per seed; accuracy uses repeat means. Table
batch costs include the batch128 tail, unlike the per-image plots.

Run `verify_report.py` after rebuilding. All four rendered pages were visually
reviewed. Regression/build evidence is recorded in `report_checks.json`.

GPU-only fitting energy excludes supervision/calibration/export/CPU costs;
payback is not a complete-system or precise cross-session uncertainty claim.
