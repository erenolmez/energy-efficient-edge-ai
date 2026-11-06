# November 6 thesis progress report

- [Report PDF](progress_report.pdf)
- [LaTeX source](progress_report.tex)
- [Official CIFAR-100 per-class test counts](class_test_run.jsonl) and [generated table](class_test_summary.tex)

Regenerate the per-class table from the published counts and checkpoint audit,
from the repository root:

```sh
python -m pc_tests.model_pipeline.analysis.build_class_summary --test
```

The [test evaluator](../../pc_tests/model_pipeline/experiments/evaluate_class_test.py)
requires the local frozen checkpoint bank and test-image cache. It is separate
from the [INT8 calibration](../../pc_tests/model_pipeline/experiments/quantize_finetuned_bank.py)
and [PC residency benchmark](../../pc_tests/model_pipeline/experiments/benchmark_finetuned_residency.py).
The benchmark code is not evidence of completed latency or energy measurements
for the new bank. Engine binaries, checkpoints, image caches, and raw telemetry
are not included. The HPC section is a W&B snapshot, not a Slurm or checkpoint audit.
