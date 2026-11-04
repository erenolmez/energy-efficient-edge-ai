# Jetson Orin Nano8GB: PC-based capacity scenarios

CPU-only arithmetic completed. **Not Jetson measurement or full hardware emulation.**

8e9-byte conservative interpretation; OS/application reserve2/3/4GiB; runtime/router0.5GiB and safety1GiB are unmeasured scenario allowances. PC allocation multipliers1/1.5/2 are sensitivity scenarios, NOT statistical bounds.

## Model residency:1.5x PC allocation scenario

Counts are prefixes of previously measured PC loading sequences, not validated Jetson maxima.

| OS/application reserve | Smallest first | Largest first | Round-robin |
|---|---:|---:|---:|
| 2GiB | 32 | 5 | 11 |
| 3GiB | 24 | 3 | 9 |
| 4GiB | 14 | 2 | 7 |

##128px image arrays:3GiB OS/application scenario

One representation at a time, except the explicit uint8+FP32 pair. Models compete with images for the SAME budget.

Two-model proxy uses the existing R34/R101 FP16+router observation,1.5x GPU-memory delta plus one serialized CPU copy.

| Classifier models | Representation | Array-capacity estimate |
|---|---|---:|
| 0 | uint8_only | 64,456 |
| 0 | fp16_only | 32,228 |
| 0 | fp32_only | 16,114 |
| 0 | uint8_plus_fp32 | 12,891 |
| 2 | uint8_only | 44,004 |
| 2 | fp16_only | 22,002 |
| 2 | fp32_only | 11,001 |
| 2 | uint8_plus_fp32 | 8,800 |

## Limitations

- No measured Jetson capacity or guaranteed safe model count; no timing/energy emulation.
- Only measured PC loading prefixes replayed, not a measured all213-target catalog.
- GPU allocation deltas are order/allocator-dependent; serialized bytes do not equal resident bytes.
- One retained serialized CPU copy is an explicit scenario, not an observed Jetson allocation.
- Image counts cover specified contiguous arrays only, no Python/PIL object overhead or extra copies.
- Temporary engine-load/build peaks and actual free Jetson RAM remain unmeasured; reserve is not proof.
- Original128px213-target models and new HPC32px models remain distinct;32px image arithmetic is not a model-capacity measurement.
- Original68GB/s and newer Super-mode102GB/s memory-bandwidth profiles do not change this arithmetic; neither is used to scale PC latency.

## Documentation

- [original_module_specs](https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/)
- [shared_memory](https://docs.pytorch.org/TensorRT/tutorials/resource_memory/resource_management.html)
- [current_module_specs](https://developer.nvidia.com/embedded/jetson-modules/)

Inputs and SHA256 identities are recorded in capacity_scenarios.json. No original results, training code, automations or hardware were changed.
