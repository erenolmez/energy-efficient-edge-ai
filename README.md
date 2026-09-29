# Energy-Efficient Edge AI

Input-aware, hardware-aware inference for resource-constrained devices.

The repository is separated by execution environment:

```text
pc_tests/
  router_benchmark/ Unified comparison of input-only routing methods
jetson_tests/       TensorRT deployment, latency, power, and router-overhead tests
tests/              Environment-independent unit tests
```

The active experiment is
[`pc_tests/router_benchmark`](pc_tests/router_benchmark). It compares scalar
difficulty, per-model correctness, handcrafted-feature, frozen-embedding, and
direct-image routers. Every router receives only the input image at runtime.
A calibration policy selects among the available configurations using measured
candidate cost and an accuracy constraint.

Neural candidate models are used offline to construct supervision labels. They
are not executed by the router before its runtime decision.

Jetson deployment experiments are intentionally isolated in
[`jetson_tests`](jetson_tests) so PC-only dependencies and TensorRT-specific code
do not become mixed.
