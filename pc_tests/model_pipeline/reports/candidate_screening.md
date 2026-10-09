# Deployment candidate screening

Each model is compared with the unpruned p0 model of the same architecture. Within each accuracy-loss budget, the selected model has the largest measured MAC reduction; parameter reduction and accuracy break ties.

| Architecture | Maximum loss | Selected model | Test accuracy | Actual loss | MAC reduction | Parameter reduction |
|---|---:|---|---:|---:|---:|---:|
| efficientnet_b0 | 0.5 pp | efficientnet_b0__p0__fp32 | 82.72% | 0.00 pp | 0.00% | 0.00% |
| efficientnet_b0 | 1.0 pp | efficientnet_b0__p0__fp32 | 82.72% | 0.00 pp | 0.00% | 0.00% |
| efficientnet_b0 | 1.5 pp | efficientnet_b0__p0__fp32 | 82.72% | 0.00 pp | 0.00% | 0.00% |
| efficientnet_b0 | 2.0 pp | efficientnet_b0__p0__fp32 | 82.72% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_large | 0.5 pp | mobilenetv3_large__p0__fp32 | 81.29% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_large | 1.0 pp | mobilenetv3_large__p0__fp32 | 81.29% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_large | 1.5 pp | mobilenetv3_large__p0__fp32 | 81.29% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_large | 2.0 pp | mobilenetv3_large__p0__fp32 | 81.29% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_small | 0.5 pp | mobilenetv3_small__p0__fp32 | 62.44% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_small | 1.0 pp | mobilenetv3_small__p0__fp32 | 62.44% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_small | 1.5 pp | mobilenetv3_small__p0__fp32 | 62.44% | 0.00 pp | 0.00% | 0.00% |
| mobilenetv3_small | 2.0 pp | mobilenetv3_small__p10__fp32 | 60.82% | 1.62 pp | 27.11% | 17.65% |
| resnet101 | 0.5 pp | resnet101__p0__fp32 | 85.80% | 0.00 pp | 0.00% | 0.00% |
| resnet101 | 1.0 pp | resnet101__p0__fp32 | 85.80% | 0.00 pp | 0.00% | 0.00% |
| resnet101 | 1.5 pp | resnet101__p0__fp32 | 85.80% | 0.00 pp | 0.00% | 0.00% |
| resnet101 | 2.0 pp | resnet101__p0__fp32 | 85.80% | 0.00 pp | 0.00% | 0.00% |
| resnet18 | 0.5 pp | resnet18__p0__fp32 | 79.16% | 0.00 pp | 0.00% | 0.00% |
| resnet18 | 1.0 pp | resnet18__p0__fp32 | 79.16% | 0.00 pp | 0.00% | 0.00% |
| resnet18 | 1.5 pp | resnet18__p0__fp32 | 79.16% | 0.00 pp | 0.00% | 0.00% |
| resnet18 | 2.0 pp | resnet18__p0__fp32 | 79.16% | 0.00 pp | 0.00% | 0.00% |
| resnet34 | 0.5 pp | resnet34__p0__fp32 | 81.80% | 0.00 pp | 0.00% | 0.00% |
| resnet34 | 1.0 pp | resnet34__p0__fp32 | 81.80% | 0.00 pp | 0.00% | 0.00% |
| resnet34 | 1.5 pp | resnet34__p0__fp32 | 81.80% | 0.00 pp | 0.00% | 0.00% |
| resnet34 | 2.0 pp | resnet34__p0__fp32 | 81.80% | 0.00 pp | 0.00% | 0.00% |
| resnet50 | 0.5 pp | resnet50__p0__fp32 | 84.34% | 0.00 pp | 0.00% | 0.00% |
| resnet50 | 1.0 pp | resnet50__p0__fp32 | 84.34% | 0.00 pp | 0.00% | 0.00% |
| resnet50 | 1.5 pp | resnet50__p0__fp32 | 84.34% | 0.00 pp | 0.00% | 0.00% |
| resnet50 | 2.0 pp | resnet50__p0__fp32 | 84.34% | 0.00 pp | 0.00% | 0.00% |
| shufflenetv2_x1_0 | 0.5 pp | shufflenetv2_x1_0__p0__fp32 | 75.63% | 0.00 pp | 0.00% | 0.00% |
| shufflenetv2_x1_0 | 1.0 pp | shufflenetv2_x1_0__p0__fp32 | 75.63% | 0.00 pp | 0.00% | 0.00% |
| shufflenetv2_x1_0 | 1.5 pp | shufflenetv2_x1_0__p0__fp32 | 75.63% | 0.00 pp | 0.00% | 0.00% |
| shufflenetv2_x1_0 | 2.0 pp | shufflenetv2_x1_0__p0__fp32 | 75.63% | 0.00 pp | 0.00% | 0.00% |
