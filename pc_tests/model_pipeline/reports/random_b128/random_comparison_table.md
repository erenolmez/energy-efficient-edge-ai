# Batch128 measured random-routing comparison

Status: COMPLETE; 495/495 saved full-test blocks.

All timings are fresh, same run and input boundary, capacity2 cold cache, all213 logically selectable; random controls do not execute the neural router. Schedule generation is offline; dispatch lookup, transfer/normalization, hash/load/eviction/inference/sync are measured. Static/learned/matched order alternates by trial. Five trials, each10000 images. Latency and GPU energy are medians over trials; accuracy is the trial mean. SD is over five randomized assignments, not independent test datasets. Energy is NVML GPU-only. Loss is versus the freshly measured R101 p0 FP32 reference; L/E savings are versus the original validation-selected eligible static identity, freshly measured for each case. Uniform has no budget or objective, and its measurements are reused across references.

| Objective | Seed | Budget pp | Policy | Accuracy mean ± SD | Loss pp | Accuracy pass (trials/5) | ms/image | L saving % | mJ/image | E saving % |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| energy | 42 | 0.5 | learned | 85.39 ± 0.00% | 0.41 | pass (5/5) | 1.463 | +6.20 | 45.016 | +6.17 |
| energy | 42 | 0.5 | matched | 85.40 ± 0.08% | 0.40 | pass (5/5) | 1.452 | +6.89 | 44.096 | +8.08 |
| energy | 42 | 0.5 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -32.84 | 55.435 | -15.55 |
| energy | 45 | 0.5 | learned | 84.51 ± 0.00% | 1.29 | FAIL (0/5) | 1.507 | +12.41 | 47.453 | +10.33 |
| energy | 45 | 0.5 | matched | 84.35 ± 0.19% | 1.45 | FAIL (0/5) | 1.253 | +27.21 | 38.606 | +27.05 |
| energy | 45 | 0.5 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -16.30 | 43.081 | +18.59 |
| energy | 48 | 0.5 | learned | 85.39 ± 0.00% | 0.41 | pass (5/5) | 1.478 | +5.31 | 44.941 | +5.92 |
| energy | 48 | 0.5 | matched | 85.39 ± 0.23% | 0.41 | pass (4/5) | 1.465 | +6.15 | 44.948 | +5.91 |
| energy | 48 | 0.5 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -17.05 | 42.980 | +10.03 |
| energy | 42 | 1 | learned | 84.51 ± 0.00% | 1.29 | FAIL (0/5) | 1.170 | +25.06 | 35.253 | +25.05 |
| energy | 42 | 1 | matched | 84.33 ± 0.12% | 1.47 | FAIL (0/5) | 1.160 | +25.72 | 34.697 | +26.23 |
| energy | 42 | 1 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -32.67 | 55.435 | -17.86 |
| energy | 45 | 1 | learned | 84.51 ± 0.00% | 1.29 | FAIL (0/5) | 1.263 | +19.15 | 38.756 | +19.30 |
| energy | 45 | 1 | matched | 84.35 ± 0.19% | 1.45 | FAIL (0/5) | 1.252 | +19.83 | 38.724 | +19.37 |
| energy | 45 | 1 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.15 | 43.081 | +10.30 |
| energy | 48 | 1 | learned | 85.12 ± 0.00% | 0.68 | pass (5/5) | 1.338 | +14.25 | 40.772 | +14.52 |
| energy | 48 | 1 | matched | 84.91 ± 0.37% | 0.89 | pass (4/5) | 1.329 | +14.87 | 40.147 | +15.83 |
| energy | 48 | 1 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -17.09 | 42.980 | +9.89 |
| energy | 42 | 1.5 | learned | 84.51 ± 0.00% | 1.29 | pass (5/5) | 0.549 | +17.25 | 23.870 | +19.89 |
| energy | 42 | 1.5 | matched | 84.33 ± 0.12% | 1.47 | pass (3/5) | 0.565 | +14.84 | 24.536 | +17.65 |
| energy | 42 | 1.5 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -212.23 | 55.435 | -86.05 |
| energy | 45 | 1.5 | learned | 83.93 ± 0.00% | 1.87 | FAIL (0/5) | 1.139 | +27.07 | 35.235 | +26.41 |
| energy | 45 | 1.5 | matched | 83.81 ± 0.20% | 1.99 | FAIL (0/5) | 1.125 | +27.95 | 34.806 | +27.31 |
| energy | 45 | 1.5 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.20 | 43.081 | +10.03 |
| energy | 48 | 1.5 | learned | 85.12 ± 0.00% | 0.68 | pass (5/5) | 1.337 | -50.66 | 40.696 | -49.74 |
| energy | 48 | 1.5 | matched | 84.91 ± 0.37% | 0.89 | pass (5/5) | 1.326 | -49.36 | 40.276 | -48.20 |
| energy | 48 | 1.5 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -105.93 | 42.980 | -58.15 |
| energy | 42 | 2 | learned | 83.08 ± 0.00% | 2.72 | FAIL (0/5) | 0.393 | +32.96 | 17.012 | +36.91 |
| energy | 42 | 2 | matched | 83.18 ± 0.17% | 2.62 | FAIL (0/5) | 0.386 | +34.16 | 16.578 | +38.52 |
| energy | 42 | 2 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -253.41 | 55.435 | -105.58 |
| energy | 45 | 2 | learned | 83.93 ± 0.00% | 1.87 | pass (5/5) | 1.138 | +27.13 | 35.132 | +26.80 |
| energy | 45 | 2 | matched | 83.81 ± 0.20% | 1.99 | pass (3/5) | 1.125 | +27.94 | 34.660 | +27.79 |
| energy | 45 | 2 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.21 | 43.081 | +10.24 |
| energy | 48 | 2 | learned | 82.29 ± 0.00% | 3.51 | FAIL (0/5) | 0.721 | +18.70 | 21.442 | +21.34 |
| energy | 48 | 2 | matched | 82.63 ± 0.19% | 3.17 | FAIL (0/5) | 0.710 | +19.95 | 21.082 | +22.66 |
| energy | 48 | 2 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -106.01 | 42.980 | -57.67 |
| latency | 42 | 0.5 | learned | 85.23 ± 0.00% | 0.57 | FAIL (0/5) | 0.627 | +4.59 | 27.674 | +6.33 |
| latency | 42 | 0.5 | matched | 85.28 ± 0.09% | 0.52 | FAIL (2/5) | 0.604 | +8.03 | 26.846 | +9.13 |
| latency | 42 | 0.5 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -215.39 | 55.435 | -87.63 |
| latency | 45 | 0.5 | learned | 84.93 ± 0.00% | 0.87 | FAIL (0/5) | 1.376 | +11.81 | 42.270 | +12.30 |
| latency | 45 | 0.5 | matched | 84.97 ± 0.12% | 0.83 | FAIL (0/5) | 1.365 | +12.50 | 42.172 | +12.50 |
| latency | 45 | 0.5 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.25 | 43.081 | +10.61 |
| latency | 48 | 0.5 | learned | 85.19 ± 0.00% | 0.61 | FAIL (0/5) | 1.421 | +8.97 | 43.079 | +9.37 |
| latency | 48 | 0.5 | matched | 85.21 ± 0.25% | 0.59 | FAIL (2/5) | 1.410 | +9.67 | 42.906 | +9.73 |
| latency | 48 | 0.5 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -17.10 | 42.980 | +9.58 |
| latency | 42 | 1 | learned | 84.02 ± 0.00% | 1.78 | FAIL (0/5) | 0.512 | +23.00 | 21.589 | +26.74 |
| latency | 42 | 1 | matched | 83.94 ± 0.15% | 1.86 | FAIL (0/5) | 0.494 | +25.57 | 21.160 | +28.20 |
| latency | 42 | 1 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -211.92 | 55.435 | -88.10 |
| latency | 45 | 1 | learned | 84.93 ± 0.00% | 0.87 | pass (5/5) | 1.375 | +12.00 | 42.161 | +12.22 |
| latency | 45 | 1 | matched | 84.97 ± 0.12% | 0.83 | pass (4/5) | 1.365 | +12.64 | 41.967 | +12.62 |
| latency | 45 | 1 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.10 | 43.081 | +10.30 |
| latency | 48 | 1 | learned | 84.94 ± 0.00% | 0.86 | pass (5/5) | 1.308 | +16.16 | 39.952 | +16.65 |
| latency | 48 | 1 | matched | 84.77 ± 0.37% | 1.03 | FAIL (2/5) | 1.297 | +16.89 | 39.530 | +17.53 |
| latency | 48 | 1 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -17.11 | 42.980 | +10.33 |
| latency | 42 | 1.5 | learned | 84.02 ± 0.00% | 1.78 | FAIL (0/5) | 0.472 | +12.02 | 21.962 | +27.28 |
| latency | 42 | 1.5 | matched | 83.94 ± 0.15% | 1.86 | FAIL (0/5) | 0.487 | +9.22 | 21.536 | +28.69 |
| latency | 42 | 1.5 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -286.23 | 55.435 | -83.55 |
| latency | 45 | 1.5 | learned | 83.93 ± 0.00% | 1.87 | FAIL (0/5) | 1.137 | +27.16 | 35.195 | +26.58 |
| latency | 45 | 1.5 | matched | 83.81 ± 0.20% | 1.99 | FAIL (0/5) | 1.126 | +27.88 | 34.775 | +27.46 |
| latency | 45 | 1.5 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.24 | 43.081 | +10.14 |
| latency | 48 | 1.5 | learned | 84.94 ± 0.00% | 0.86 | pass (5/5) | 1.308 | -47.46 | 39.828 | -47.27 |
| latency | 48 | 1.5 | matched | 84.77 ± 0.37% | 1.03 | pass (4/5) | 1.297 | -46.20 | 39.425 | -45.78 |
| latency | 48 | 1.5 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -105.98 | 42.980 | -58.93 |
| latency | 42 | 2 | learned | 84.02 ± 0.00% | 1.78 | pass (5/5) | 1.059 | +32.21 | 32.083 | +33.04 |
| latency | 42 | 2 | matched | 83.94 ± 0.15% | 1.86 | pass (4/5) | 1.047 | +32.92 | 31.665 | +33.92 |
| latency | 42 | 2 | uniform | 36.15 ± 2.07% | 49.65 | FAIL (0/5) | 2.072 | -32.70 | 55.435 | -15.69 |
| latency | 45 | 2 | learned | 83.93 ± 0.00% | 1.87 | pass (5/5) | 1.390 | +11.08 | 42.853 | +11.57 |
| latency | 45 | 2 | matched | 83.81 ± 0.20% | 1.99 | pass (3/5) | 1.197 | +23.42 | 37.050 | +23.54 |
| latency | 45 | 2 | uniform | 37.28 ± 4.22% | 48.52 | FAIL (0/5) | 2.002 | -28.02 | 43.081 | +11.10 |
| latency | 48 | 2 | learned | 82.25 ± 0.00% | 3.55 | FAIL (0/5) | 0.668 | +24.71 | 19.934 | +27.49 |
| latency | 48 | 2 | matched | 82.44 ± 0.16% | 3.36 | FAIL (0/5) | 0.659 | +25.81 | 20.060 | +27.04 |
| latency | 48 | 2 | uniform | 35.34 ± 2.57% | 50.46 | FAIL (0/5) | 1.828 | -105.88 | 42.980 | -56.33 |
