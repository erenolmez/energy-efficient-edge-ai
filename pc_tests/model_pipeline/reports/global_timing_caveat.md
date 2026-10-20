# Interim global comparison caveat — 11 October 2026, 17:59 UTC

Six global training groups are complete. The checkpoint inspected here contains
16 completed routed configurations, all satisfying their held-out accuracy
budgets. This is not completion of the 1,440-case matrix.

The batch-64 energy/seed-42 gradient-stump policies select unpruned ResNet-101
FP16 for every held-out image. Their comparator is the same classifier. Both
paths record one classifier engine load per repeat. Consequently their reported
positive savings cannot establish an advantage from dynamic model selection.

For the 0.5-pp policy, five routed 10,000-image pass durations were approximately
22.00, 22.04, 15.21, 14.58 and 14.59 seconds. Its freshly preceding comparator
passes were approximately 21.26, 21.24, 21.32, 21.28 and 21.34 seconds. The
1.5-pp policy subsequently measured stable routed passes around 14.56–14.66
seconds and static passes around 14.11–14.19 seconds, with 3.21% more routed
energy. Sequential measurements therefore exhibit substantial nonstationarity;
the source of that variability has not been established.

Retain the raw rows, including positive and negative savings. Do not describe
the 0.5/1.0-pp apparent wins as proof of routing feasibility or select policies
using held-out outcomes. An independent, explicitly recorded interleaved
comparison with GPU clocks/utilization and host-load diagnostics is needed
before making a robust superiority claim. No active measurement process was
interrupted or modified during this inspection.
