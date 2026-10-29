# Progress report, 29 October 2026

`progress_report.tex` is the editable three-page LaTeX report; `progress_report.pdf` is the compiled
presentation. The 13 and 20 October reports are preserved unchanged.

The report now shows 1pp and 2pp accuracy-pass counts side by side for the same
frozen policies. All 15 compact-router cases meet the looser 2pp gate; four also
record positive latency and GPU-energy savings against unchanged 1pp-selected
static controls. This is a descriptive sensitivity check, not new 2pp calibration
or a rerun. Historical result claims and payback retain their original gates.

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
payback. It includes larger-batch results, the compact batch1 comparison and
switching/residency/dispatch investigations. The training/per-image batching
section is removed from the three-page presentation; its existing scientific
data and supplementary assets are preserved, not deleted. Historical
213-choice/capacity2 and new six-resident-model protocols
are explicitly distinguished. The cancelled full batch1 matrix is not claimed
complete. No inference, retraining or automation is started by these scripts.

The first accuracy/latency/energy figure now includes both fresh static baselines
at every batch size: eligible static and R101 FP32, alongside learned TinyCNN.
All three series use the same seed/case selection; median markers and seed-range
bars are descriptive, not confidence intervals. Open batch1 markers denote the
different six-resident eager setup. The frozen snapshot includes all36 baseline
rows, and the report verifier checks their coverage and batch1 values.

Rebuild assets with `build_report_assets.py`, then normalize generated prose
using `format_report.py`. Compile `progress_report.tex` with Tectonic or a compatible TeX
distribution. Required images and generated TeX tables are supplied so PDF
compilation alone does not need datasets, engines or measurement artifacts.
`report_data.json` freezes report inputs and their SHA256 hashes. Historical
five-repeat costs use medians per seed; accuracy uses repeat means. Table
batch costs include the batch128 tail, unlike the per-image plots.

Runtime modes are explained in five numbered paragraphs rather than a compressed
table. The payback table separates the number of settings that recover fitting
energy from the number of images processed before repayment, with a plain-English
reading example; its accuracy-pass column is removed without changing the
scientific eligibility rules or numerical results.

Run `verify_report.py` after rebuilding. All three current pages were visually checked.
Regression/build evidence is recorded in `report_checks.json`.

GPU-only fitting energy excludes supervision/calibration/export/CPU costs;
payback is not a complete-system or precise cross-session uncertainty claim.
