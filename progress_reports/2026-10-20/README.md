# Thesis progress report: 20 October 2026

Compile `progress_report.tex` with a TeX distribution supporting TikZ and the
packages declared in its preamble. The checked PDF has five pages.

The presentation includes scalar/correctness router mechanisms, the changing
1.5-point comparator, an unimplemented model-profile shared-scorer proposal and
related literature, measured batch64/batch128 learned versus uniform random
comparisons, fixed p90 static controls, and unpruned FP32 class statistics.

At the user's final request, section 5 uses the original measured **p90** controls,
not relabelled p80 measurements. The completed p80 study and `p80_savings.tex`
remain separate evidence and are not the table shown in the final presentation.
No measurements were rerun for this revision. Both studies' accuracy failures
are retained; cost savings do not imply meeting the accuracy constraint.

Required TeX includes: `tables.tex`, `uniform_batches_savings.tex`,
`pruned_savings.tex`, and `class_statistics_section.tex`.
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
