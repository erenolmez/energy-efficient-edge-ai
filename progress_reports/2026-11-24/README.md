# 24 November thesis progress report

- [Report PDF](progress_report.pdf) and [LaTeX source](progress_report.tex).
- [Routed benchmark](../../pc_tests/model_pipeline/experiments/run_meeting_routed_plateau.py), [measurement audit](../../pc_tests/model_pipeline/analysis/audit_optimized_routed_plateau.py), and [report table builder](../../pc_tests/model_pipeline/analysis/build_routed_resnet101_rows.py).
- [ImageNet comparison trainer](../../hpc/train_imagenet_comparison.py), [held-out evaluator](../../hpc/evaluate_imagenet_comparison.py), and [scratch test evaluator](../../pc_tests/model_pipeline/experiments/evaluate_meeting_scratch32.py).
- [Scratch structural pruning, fine-tuning, and per-class evaluation](../../hpc/prune_scratch_comparison.py).

Build from this directory with `tectonic progress_report.tex`. The PDF includes the
16 completed pretrained evaluations and the 63 completed pruning cases. The
pretrained and pruning evaluation entry points package the workflow used on
Slurm; the published tables reflect those cluster runs, not a local rerun.

HPC entry points are run from the repository root in an allocated GPU job.
The ImageNet evaluator takes the study parent, CIFAR-100 root, and a new output
directory. The pruning entry point additionally takes the frozen scratch case
and requested pruning percentage (10 through 90). Each pruning level starts
from the unpruned, validation-selected checkpoint. The evaluators do not
download CIFAR-100, and this publication does not include checkpoints, engine
binaries, image caches, or raw telemetry.
