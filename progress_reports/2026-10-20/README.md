# Thesis progress report: 20 October 2026

Compile `progress_report.tex` with a TeX distribution supporting TikZ and the
packages declared in its preamble. The checked PDF has five pages.

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
