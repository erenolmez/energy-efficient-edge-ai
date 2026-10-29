# Paired dedicated batch-one engine profiles

Five paired >=2s validation-input windows; new profiles may change logits/argmax. No full-test routing gain, no whole-PC energy or unobserved peak-memory claim.

| Model | Original ms/image | Batch1 ms/image | Latency saving | GPU-energy saving | Argmax changes /100 |
|---|---:|---:|---:|---:|---:|
| resnet101__p0__fp32 | 6.132 | 4.374 | +28.7% | +28.8% | 0 |
| resnet101__p0__fp16 | 4.290 | 2.616 | +39.0% | +34.1% | 1 |
| efficientnet_b0__p0__fp16 | 2.851 | 2.759 | +3.2% | +5.0% | 0 |
| resnet34__p0__int8 | 1.570 | 1.399 | +10.9% | +8.6% | 40 |
| mobilenetv3_large__p10__fp16 | 1.991 | 1.934 | +2.9% | +4.9% | 1 |
| resnet18__p10__fp16 | 1.169 | 0.912 | +21.9% | +19.5% | 1 |

Same6models held resident:
- original: observed incremental 1178 MiB; 3847 MiB free.
- batch1: observed incremental 316 MiB; 4709 MiB free.

Selection used seed42fit positions only. Preserve all negative savings and discrepancies. Dedicated-engine validation correctness must be refreshed before policy calibration.

IMPORTANT: ResNet34 p0 INT8 changed40/100probeclasses. Do not treat that engine as numerically interchangeable or count its speed change as equivalent-model optimization; diagnose before downstream use. Structural audit passed does not mean numerical acceptance.
