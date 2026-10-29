# Code and evidence for the October29 report

## Published material

- `progress_report.tex`, included numerical tables, report figures and the compiled
  three-page `progress_report.pdf`; October13/20 sources are referenced, not replaced.
- PC experiment implementations under `pc_tests/model_pipeline/experiments/`:
  training-energy measurement, residency probing, dedicated-engine profiles and
  INT8 diagnosis, compact-router preparation and inference, switching pilots and
  confirmation, GPU dispatch/conditional probes, training/calibration screens,
  per-image batch8 grouping, and final supervision deployment.
- Supporting modules in `runtime/` and `routers/correctness/`, CPU audits and
  summaries in `analysis/`, and guarded local launchers in `scripts/`.
- Thirteen targeted test modules, with38 CPU tests passing. The two-iteration
  synthetic MLP convergence/batch-size warnings are retained, not test failures.
- Small audited JSON/CSV/Markdown results, historical larger-batch payback
  summaries and scientific figures. `code_manifest.json` lists exact files,
  sizes and SHA256 hashes; it does not overwrite historical provenance.

## Measurement boundaries

Study A compares four CPU routers and the original TinyCNN with six resident
classifiers and eager execution. Study B implements runtime optimization in
full TinyCNN routing runs,81 sweeps with three seeds and three repeats. The
conditional/GPU-selection/profile probes are separate engineering tests, not
an all-optimizations-combined deployment result. The cancelled multi-day full
batch1 matrix remains cancelled; its small stop receipt is included to prevent
misinterpreting the historical running flag as authority to resume it.

GPU-only energy is not CPU/system energy. Payback requires the corresponding
fit energy, positive comparator-minus-routed inference saving and acceptable
held-out accuracy. Original source/report hashes, failures, negative savings,
static collapse and the quarantined dedicated INT8 build remain explicit.

## Reproduction boundaries and safety

The supplied TeX and images compile without any engines, weights or datasets.
Asset regeneration and full provenance verification need the measurement
inputs referenced by the scripts; absolute local paths in historical hashes
identify the original experiment environment, not portable file locations.
Raw checkpoints/reports, prediction arrays, engine binaries, fitted weights,
dataset caches, machine pause/shutdown receipts and local dependency bundles
are intentionally excluded. This is code and compact evidence publication,
not a complete measurement archive or a claim that every audit can run in a
fresh clone without those retained inputs.

Python components use NumPy, PyTorch, scikit-learn and ONNX; GPU runs additionally
require the existing TensorRT/NVML/CUDA stack described in the pipeline's
requirements files. Report plotting/verifying uses Matplotlib and pypdf. The
measured environment is retained; this publication does not install or change it.

The PowerShell launchers are historical, machine-specific workflow code.
They launch hidden processes with exact-identity checks and thermal guards;
the guard can request a graceful PC shutdown after its thermal latch. Do not
execute them as generic examples, overlap workers or revive a cancelled study.
Publishing or importing source does not authorize new experiments. No GPU run,
HPC submission, hardware change or shutdown is part of this publication.

Unrelated Jetson/HPC files, root documentation and edits to the older report
are left outside this commit.
