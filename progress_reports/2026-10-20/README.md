# Thesis progress report: 20 October 2026

Compile `progress_report.tex` with a TeX distribution supporting TikZ and the
packages declared in its preamble. The checked PDF has four pages. The revised
first page explicitly explains correctness targets, training, outputs, batch
aggregation, model selection and accuracy-allowance calibration. The ML versus
optimization section is condensed; the adaptable-router proposal occupies
approximately half a page.

The presentation includes scalar/correctness router mechanisms, the changing
1.5-point comparator, an unimplemented model-profile shared-scorer proposal and
related literature, measured batch64/batch128 learned versus uniform random
comparisons, fixed p80 static controls, and unpruned FP32 class statistics.

Section 5 uses the completed, audited **p80** measurements, replacing the earlier
p90 presentation table. Its 440 full-test sweeps cover batches8/32/64/128,
seven architectures and three precisions. Accuracy, latency and energy values
come from the p80 study; no values have been relabelled from p90. Historical
p90 files remain preserved. Cost savings do not imply accuracy compliance.

Required TeX includes: `tables.tex`, `uniform_batches_savings.tex`,
`p80_savings.tex`, and `class_statistics_section.tex`.
`render_p80_table.py` generates the p80 table from the audited 84-row summary.
The p80 summary, numerical table and completion audit are published under
`pc_tests/model_pipeline/reports/p80_b8_32_64_128/`.

`numerical_snapshot.json` preserves the original results and source hashes;
its partial random64 capture is historical, not the final random64 result.
The finished batch64 table uses the audited final summary, with its source hash
and original-accuracy checks recorded in `uniform_batches_table_audit.json`.
`allowance_change_audit.json` records the 1.5-point comparator investigation.
The separate `all_213_class_table` files are a 71-row supplementary table.

`render_uniform_batches.py` documents the combined table generation;
`render_savings_tables.py` documents the original p90 and uniform table generation.
These scripts require the corresponding local measurement reports. The frozen
TeX includes are supplied so PDF compilation does not require engines, weights,
datasets, GPU access, or those measurement files. Temporary render/build files
are excluded from version control.

## Code and evidence map

Experiment code stays in `pc_tests/model_pipeline/`; unit tests stay in `tests/`.
The report-specific generators and frozen TeX tables stay in this folder.
No separate copy of the experiment implementation is maintained here.

| Report material | Implementation | Evidence / table generation |
|---|---|---|
| Scalar routing controls | `train_pc_batch_routers.py`, `batch_router_policy.py`, `gpu_router_models.py`, `run_pc_routes.py` | Historical results in `numerical_snapshot.json`; scalar runner retained as a control, not authorized to resume |
| Correctness routing | `correctness_router.py`, `correctness_policy.py`, `train_correctness_router.py`, `run_correctness_routes.py` | Historical matrix and hashes in `numerical_snapshot.json`; 1.5-point investigation: `audit_allowance_changes.py` and `allowance_change_audit.json` |
| Learned versus uniform random, batches 64/128 | `run_random_routes_b64.py`, `run_random_routes.py`, `random_route_baselines.py`; corresponding `summarize_random_routes*.py` | `reports/random_b64/` and `reports/random_b128/` summary JSON/Markdown; `render_uniform_batches.py` produces `uniform_batches_savings.tex` |
| Fixed p80 controls | `run_p80_batches.py`, `audit_p80_batches.py`, `test_p80_batches.py` | `reports/p80_b8_32_64_128/` summary, table and completion audit; `render_p80_table.py` produces `p80_savings.tex` |
| Historical p90 controls | `run_most_pruned.py`, `run_most_pruned_batches.py` | Small summaries/tables in `reports/most_pruned_b128/` and `reports/most_pruned_b8_32_64/`; these are not the current section 5 |
| Per-class accuracy | `summarize_fp32_classes.py`, `summarize_all_class_statistics.py` | `reports/class_statistics/`; frozen `tables.tex` and separate all-213 supplement |
| Runtime boundary | `gpu_batch_inputs.py`, `global_candidate_pool.py`, `gpu_engine_preflight.py`, `gpu_energy.py`, `router_tensorrt.py` | Shared preprocessing, engine loading/cache, inference/synchronization and GPU-energy helpers |

All implementation filenames above are relative to `pc_tests/model_pipeline/`
unless explicitly described as report-local. `build_snapshot.py` documents the
original report snapshot generation; do not run it to reproduce the frozen
presentation, because it reads current local experiment state. Some additional
correctness router families are source implementations only, not completed
comparative measurements in this report. The adaptable profile scorer and Deep
Sets discussion remains a proposal; no implementation or measured result is
claimed for it.

### Reproduction boundaries

The published small summaries and TeX tables reproduce the presentation. Full
experiment replay additionally needs the original deployment manifest, weights,
TensorRT engines, dataset/image caches, prediction arrays, runtime versions and
raw per-sweep local reports. Those large/local artifacts, temporary files and
recovery backups are not published in this commit. Source/report hashes identify
the measured inputs; publishing code is not evidence that new runs occurred.

With those existing local inputs, CPU-only audits can be run from the repository
root (they inspect saved results; they do not launch inference):

```powershell
.\.cuda-venv\Scripts\python.exe pc_tests/model_pipeline/audit_random_b64.py
.\.cuda-venv\Scripts\python.exe pc_tests/model_pipeline/audit_p80_batches.py
```

`start_p80_batches.ps1`, `start_most_pruned.ps1` and `watch_pc_thermals.ps1` are
historical local launch/thermal-guard helpers, not portable one-click examples.
They contain machine-specific paths and can stop verified experiment processes
and request graceful shutdown after a thermal latch. Never start a runner without
its matching guard, overlap studies, or restart a latched/stopped study. No new
experiments or monitoring are started by this publication; Jetson remains paused.
Requirements are in the pipeline's `requirements.txt` and `requirements-gpu.txt`;
CUDA PyTorch and the existing router dependencies must be supplied separately.

Publication verification: 54 targeted unit tests passed with CUDA devices hidden.
`code_manifest.json` records the published source/test hashes. This CPU check
does not establish GPU timing, export parity or performance claims beyond the
already audited measurements.
