# Batch-size comparison and fit-energy payback

GPU-only fit-energy repeat, not full preparation/whole-PC; historical inference and new training/b1 are separate sessions with nonstationary GPU conditions. No test-label tuning or retrospective best-seed selection. Seed variation mixes training/split/batching. No precise statistical confidence claim.

Payback images = ceil(seed-specific training J / (static J/image - routed J/image)). Calculate independently for every batch, seed, objective, allowance and comparator. Null valid payback is retained for accuracy failures or nonpositive savings. No three-seed cost is charged to one deployed router.

| Batch | Settings | Accuracy passes | Positive energy saving vs eligible static | Valid payback range (images) |
|---|---:|---:|---:|---:|
| 8 | 24 | 10 | 2 | 350,497 to 1,314,076 |
| 16 | 24 | 8 | 10 | 79,015 to 81,556 |
| 32 | 24 | 13 | 17 | 29,522 to 1,672,462 |
| 64 | 24 | 11 | 21 | 9,583 to 23,494,940 |
| 128 | 24 | 11 | 22 | 16,089 to 96,831 |

Plots show all settings or a fixed1pp allowance, not the most favorable test-selected configuration. Detailed CSV/JSON retain every seed, negative and failed accuracy case.

Full-batch payback in batches is ceil(training J / (batch size x saved J/image)); image counts in the JSON are continuous-throughput estimates rounded up, not padded batch counts.
