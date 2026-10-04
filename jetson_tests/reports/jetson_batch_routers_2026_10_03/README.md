# Batch-router measurements on Jetson Orin Nano

## Experiment

These measurements compare one-decision-per-batch routers with the unpruned
FP32 ResNet-18 (`p0`) baseline. No oracle result is used as a baseline.

- Candidate inference models: FP32 `p0`, `p10`, `p20`, and `p30`.
- Routers: gradient stump, depth-3 tree, logistic regression, depth-2 XGBoost,
  PyTorch TinyCNN, and TensorRT FP16 TinyCNN.
- Batch sizes: 32, 64, and 128.
- Accuracy-loss limits: 0.5, 1.0, 1.5, and 2.0 percentage points.
- Evaluation data: the same fixed, stratified 2,000 CIFAR-100 test images for
  every method.
- Repeats: three complete system measurements per configuration.
- Energy run: GPU clock fixed at 624.75 MHz.
- Latency run: GPU clock fixed at 612 MHz.

The router chooses one inference model for the complete batch. Every reported
latency and energy value includes input analysis, router execution, selected
model inference, transfers, and synchronization. Energy is calculated from
the integrated `VDD_IN` measurement and reported as joules per image.

The `p0` accuracy on this fixed 2,000-image subset is 72.55%. This is not a new
full-test-set accuracy measurement; the earlier full 10,000-image TensorRT
result remains a separate experiment.

## Strict 0.5-point results

Best energy-policy router at each batch size:

| Batch | Router | Accuracy | Accuracy vs p0 | Energy saving | Latency saving |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | Logistic | 72.55% | 0.00 pp | 2.45% | -2.75% |
| 64 | TinyCNN TensorRT | 73.30% | +0.75 pp | 0.07% | -2.24% |
| 128 | Logistic | 72.65% | +0.10 pp | 4.90% | 2.74% |

Best latency-policy router at each batch size:

| Batch | Router | Accuracy | Accuracy vs p0 | Latency saving | Energy saving |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | Logistic | 72.55% | 0.00 pp | -2.17% | 2.83% |
| 64 | TinyCNN TensorRT | 73.30% | +0.75 pp | -2.68% | -0.57% |
| 128 | Logistic | 72.65% | +0.10 pp | 1.16% | 4.05% |

A negative saving means the routed system is slower or consumes more energy
than running `p0` directly.

For comparison, the static `p10` model stays within the same 0.5-point limit:

| Batch | Accuracy | Accuracy vs p0 | Energy saving | Latency saving |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 72.50% | -0.05 pp | 10.13% | 7.98% |
| 64 | 72.50% | -0.05 pp | 9.30% | 7.41% |
| 128 | 72.50% | -0.05 pp | 8.80% | 7.16% |

The static `p10` deployment is therefore the strongest choice under the strict
0.5-point requirement. The dynamic router does not recover enough model cost
to repay its own overhead, especially at batches 32 and 64.

## Relaxed accuracy limits

Routing becomes more competitive when the allowed loss increases. At batch
128, the energy policy reaches 19.29%, 21.01%, and 22.80% energy savings for
the 1.0, 1.5, and 2.0-point limits. The corresponding best latency-policy
savings are 14.59%, 16.11%, and 17.07%.

Static models still win most comparisons. The main exception is energy at the
1.5-point limit: the best router saves 19.24% at batch 64 and 21.01% at batch
128, compared with 18.48% and 19.31% for the best eligible static model. This
advantage is modest and does not appear in the latency objective. At the
2.0-point limit, static `p30` is clearly stronger, saving up to 32.62% energy
and 29.06% latency at batch 128.

## Model selections and overhead

Under the strict policy, the chosen router sends only some batches to a pruned
model:

| Objective | Batch | p0 images | p10 images | p20 images | p30 images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Energy | 32 | 1,008 | 0 | 992 | 0 |
| Energy | 64 | 1,152 | 848 | 0 | 0 |
| Energy | 128 | 1,280 | 0 | 720 | 0 |
| Latency | 32 | 1,008 | 0 | 992 | 0 |
| Latency | 64 | 1,152 | 848 | 0 | 0 |
| Latency | 128 | 1,280 | 0 | 720 | 0 |

The best strict routers spend about 5.1 ms per batch at batch 32, 6.0 ms at
batch 64, and 11.1 ms at batch 128. The per-image overhead is amortized best at
batch 128. TensorRT lowers TinyCNN overhead relative to its PyTorch version,
but model choice quality still determines whether the total system saves cost.

Accuracy can be slightly higher than `p0` because the pruned and unpruned
models make different errors on the same images. This is an observed routed
system result, not an oracle upper bound.

## Conclusion

For the current ResNet-18 pruning family, use static `p10` when the maximum
accuracy loss is 0.5 percentage points. Keep dynamic routing as a research
result rather than the recommended deployment at that constraint. Its clearest
benefit is a small energy advantage at the 1.5-point limit for batches 64 and
128.

The present candidates make similar errors and have limited cost separation.
The next experiment will therefore add architecture and precision diversity
before increasing router capacity. Small-batch deployment also requires a
fused GPU or TensorRT feature path.

## Files

- `energy_all_routes.csv` and `latency_all_routes.csv`: every measured router.
- `energy_best_routes.csv` and `latency_best_routes.csv`: best valid router for
  each batch size and accuracy limit.
- `energy_best_static.csv` and `latency_best_static.csv`: best eligible static
  model under the same limits.
- `static_energy_summary.csv` and `static_latency_summary.csv`: all static
  model measurements.
- `best_router_vs_static.png`: direct static-versus-routing comparison.
- `strict_policy_model_selections.png`: selected-model counts at 0.5 pp.
- `router_overhead_by_method.png`: measured router time per batch.
- `best_savings_by_accuracy_budget.png`: routed-system trade-off curves.
