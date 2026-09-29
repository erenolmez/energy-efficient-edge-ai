# Jetson tests

This directory is reserved for deployment-side experiments and will remain
separate from the current PC workflow.

Planned components:

```text
engines/       Local TensorRT engines (never committed)
manifests/     Engine IDs, precision, pruning, and deployment metadata
scripts/       TensorRT prediction and router-overhead measurement
results/       Latency, power, energy, accuracy, and routing results
```

Later experiments will cover:

1. TensorRT per-image prediction collection using the same CSV interface as the
   PC pipeline.
2. Router feature-extraction and XGBoost latency/energy measurement.
3. End-to-end dynamic execution with engine selection.
4. Tiny-CNN and confidence-routing baselines.
5. Static p0 and pruned-model comparisons using measured Jetson energy.

TensorRT engines and large result artifacts should stay local. Small manifests,
scripts, aggregated CSVs, and documentation can be version controlled.
