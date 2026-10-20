# Batch64 measured random-routing comparison

Status: COMPLETE; 495/495 saved full-test blocks.

All timings are fresh, same run and input boundary, capacity2 cold cache, all213 logically selectable; random controls do not execute the neural router. Schedule generation is offline; dispatch lookup, transfer/normalization, hash/load/eviction/inference/sync are measured. Static/learned/matched order alternates by trial. Five trials, each10000 images. Latency and GPU energy are medians over trials; accuracy is the trial mean. SD is over five randomized assignments, not independent test datasets. Energy is NVML GPU-only. Loss is versus the freshly measured R101 p0 FP32 reference; L/E savings are versus the original validation-selected eligible static identity, freshly measured for each case. Uniform has no budget or objective, and its measurements are reused across references.

| Objective | Seed | Budget pp | Policy | Accuracy mean ± SD | Loss pp | Accuracy pass (trials/5) | ms/image | L saving % | mJ/image | E saving % |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| energy | 42 | 0.5 | learned | 85.23 ± 0.00% | 0.57 | FAIL (0/5) | 1.435 | +4.23 | 43.792 | +3.97 |
| energy | 42 | 0.5 | matched | 85.18 ± 0.19% | 0.62 | FAIL (1/5) | 1.388 | +7.36 | 42.246 | +7.36 |
| energy | 42 | 0.5 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -126.72 | 68.938 | -51.17 |
| energy | 45 | 0.5 | learned | 85.27 ± 0.00% | 0.53 | FAIL (0/5) | 1.313 | +6.36 | 40.214 | +5.93 |
| energy | 45 | 0.5 | matched | 85.24 ± 0.21% | 0.56 | FAIL (2/5) | 1.299 | +7.34 | 39.689 | +7.16 |
| energy | 45 | 0.5 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -149.22 | 77.015 | -80.15 |
| energy | 48 | 0.5 | learned | 85.64 ± 0.00% | 0.16 | pass (5/5) | 1.403 | +3.77 | 43.017 | +3.58 |
| energy | 48 | 0.5 | matched | 85.67 ± 0.09% | 0.13 | pass (5/5) | 1.460 | -0.17 | 44.735 | -0.27 |
| energy | 48 | 0.5 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -139.69 | 86.202 | -93.21 |
| energy | 42 | 1 | learned | 84.44 ± 0.00% | 1.36 | FAIL (0/5) | 1.161 | +24.03 | 34.422 | +24.11 |
| energy | 42 | 1 | matched | 84.46 ± 0.18% | 1.34 | FAIL (0/5) | 1.195 | +21.81 | 35.454 | +21.84 |
| energy | 42 | 1 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -122.23 | 68.938 | -51.98 |
| energy | 45 | 1 | learned | 84.74 ± 0.00% | 1.06 | FAIL (0/5) | 1.258 | +9.37 | 38.516 | +9.34 |
| energy | 45 | 1 | matched | 85.00 ± 0.19% | 0.80 | pass (4/5) | 1.241 | +10.58 | 37.762 | +11.11 |
| energy | 45 | 1 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -151.82 | 77.015 | -81.28 |
| energy | 48 | 1 | learned | 85.18 ± 0.00% | 0.62 | pass (5/5) | 1.379 | +8.35 | 42.125 | +7.72 |
| energy | 48 | 1 | matched | 85.28 ± 0.05% | 0.52 | pass (5/5) | 1.330 | +11.64 | 40.412 | +11.47 |
| energy | 48 | 1 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -132.17 | 86.202 | -88.83 |
| energy | 42 | 1.5 | learned | 84.44 ± 0.00% | 1.36 | pass (5/5) | 0.559 | +19.91 | 24.517 | +19.85 |
| energy | 42 | 1.5 | matched | 84.46 ± 0.18% | 1.34 | pass (5/5) | 0.557 | +20.16 | 24.146 | +21.06 |
| energy | 42 | 1.5 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -386.63 | 68.938 | -125.37 |
| energy | 45 | 1.5 | learned | 84.34 ± 0.00% | 1.46 | pass (5/5) | 1.173 | +17.44 | 35.959 | +15.93 |
| energy | 45 | 1.5 | matched | 84.69 ± 0.20% | 1.11 | pass (5/5) | 1.158 | +18.48 | 35.289 | +17.50 |
| energy | 45 | 1.5 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -145.97 | 77.015 | -80.05 |
| energy | 48 | 1.5 | learned | 84.66 ± 0.00% | 1.14 | pass (5/5) | 1.253 | -44.79 | 37.976 | -42.94 |
| energy | 48 | 1.5 | matched | 84.79 ± 0.14% | 1.01 | pass (5/5) | 1.218 | -40.76 | 36.171 | -36.14 |
| energy | 48 | 1.5 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -303.85 | 86.202 | -224.45 |
| energy | 42 | 2 | learned | 83.51 ± 0.00% | 2.29 | FAIL (0/5) | 0.716 | -35.39 | 27.902 | -8.84 |
| energy | 42 | 2 | matched | 83.50 ± 0.12% | 2.30 | FAIL (0/5) | 0.715 | -35.24 | 29.117 | -13.58 |
| energy | 42 | 2 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -542.44 | 68.938 | -168.91 |
| energy | 45 | 2 | learned | 83.49 ± 0.00% | 2.31 | FAIL (0/5) | 0.983 | +29.01 | 30.315 | +28.98 |
| energy | 45 | 2 | matched | 83.47 ± 0.27% | 2.33 | FAIL (0/5) | 0.970 | +29.92 | 30.050 | +29.60 |
| energy | 45 | 2 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -152.36 | 77.015 | -80.42 |
| energy | 48 | 2 | learned | 83.65 ± 0.00% | 2.15 | FAIL (0/5) | 1.000 | -15.65 | 29.824 | -13.07 |
| energy | 48 | 2 | matched | 83.86 ± 0.19% | 1.94 | pass (3/5) | 1.037 | -19.97 | 31.396 | -19.04 |
| energy | 48 | 2 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -304.22 | 86.202 | -226.83 |
| latency | 42 | 0.5 | learned | 84.93 ± 0.00% | 0.87 | FAIL (0/5) | 0.502 | +10.53 | 23.033 | +12.60 |
| latency | 42 | 0.5 | matched | 84.81 ± 0.20% | 0.99 | FAIL (0/5) | 0.507 | +9.57 | 22.996 | +12.75 |
| latency | 42 | 0.5 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -505.83 | 68.938 | -161.57 |
| latency | 45 | 0.5 | learned | 85.52 ± 0.00% | 0.28 | pass (5/5) | 1.352 | +2.73 | 41.406 | +2.57 |
| latency | 45 | 0.5 | matched | 85.50 ± 0.11% | 0.30 | pass (5/5) | 1.338 | +3.74 | 41.095 | +3.30 |
| latency | 45 | 0.5 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -151.44 | 77.015 | -81.22 |
| latency | 48 | 0.5 | learned | 85.41 ± 0.00% | 0.39 | pass (5/5) | 1.388 | +3.59 | 42.222 | +5.28 |
| latency | 48 | 0.5 | matched | 85.41 ± 0.06% | 0.39 | pass (5/5) | 1.303 | +9.48 | 40.193 | +9.83 |
| latency | 48 | 0.5 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -142.76 | 86.202 | -93.39 |
| latency | 42 | 1 | learned | 84.02 ± 0.00% | 1.78 | FAIL (0/5) | 0.434 | +19.54 | 19.845 | +22.28 |
| latency | 42 | 1 | matched | 83.98 ± 0.19% | 1.82 | FAIL (0/5) | 0.423 | +21.62 | 19.467 | +23.76 |
| latency | 42 | 1 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -529.41 | 68.938 | -169.99 |
| latency | 45 | 1 | learned | 85.17 ± 0.00% | 0.63 | pass (5/5) | 1.292 | +6.67 | 39.571 | +6.28 |
| latency | 45 | 1 | matched | 85.16 ± 0.25% | 0.64 | pass (5/5) | 1.277 | +7.72 | 38.926 | +7.81 |
| latency | 45 | 1 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -152.46 | 77.015 | -82.39 |
| latency | 48 | 1 | learned | 84.84 ± 0.00% | 0.96 | pass (5/5) | 1.226 | +12.64 | 36.870 | +14.86 |
| latency | 48 | 1 | matched | 85.03 ± 0.14% | 0.77 | pass (5/5) | 1.213 | +13.55 | 36.849 | +14.90 |
| latency | 48 | 1 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -149.03 | 86.202 | -99.07 |
| latency | 42 | 1.5 | learned | 84.02 ± 0.00% | 1.78 | FAIL (0/5) | 0.417 | +19.45 | 20.351 | +20.76 |
| latency | 42 | 1.5 | matched | 83.98 ± 0.19% | 1.82 | FAIL (0/5) | 0.411 | +20.70 | 19.547 | +23.89 |
| latency | 42 | 1.5 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -555.50 | 68.938 | -168.40 |
| latency | 45 | 1.5 | learned | 84.58 ± 0.00% | 1.22 | pass (5/5) | 1.229 | +11.27 | 37.423 | +11.19 |
| latency | 45 | 1.5 | matched | 84.94 ± 0.25% | 0.86 | pass (5/5) | 1.213 | +12.41 | 37.213 | +11.69 |
| latency | 45 | 1.5 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -152.21 | 77.015 | -82.76 |
| latency | 48 | 1.5 | learned | 84.84 ± 0.00% | 0.96 | pass (5/5) | 1.228 | -45.04 | 37.301 | -44.13 |
| latency | 48 | 1.5 | matched | 85.03 ± 0.14% | 0.77 | pass (5/5) | 1.214 | -43.41 | 37.048 | -43.15 |
| latency | 48 | 1.5 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -312.71 | 86.202 | -233.08 |
| latency | 42 | 2 | learned | 83.08 ± 0.00% | 2.72 | FAIL (0/5) | 0.823 | +45.00 | 24.731 | +45.90 |
| latency | 42 | 2 | matched | 83.07 ± 0.15% | 2.73 | FAIL (0/5) | 0.787 | +47.42 | 23.663 | +48.23 |
| latency | 42 | 2 | uniform | 36.36 ± 2.64% | 49.44 | FAIL (0/5) | 3.397 | -127.04 | 68.938 | -50.81 |
| latency | 45 | 2 | learned | 83.23 ± 0.00% | 2.57 | FAIL (0/5) | 0.962 | +34.77 | 29.709 | +34.51 |
| latency | 45 | 2 | matched | 83.40 ± 0.20% | 2.40 | FAIL (0/5) | 0.967 | +34.43 | 30.446 | +32.89 |
| latency | 45 | 2 | uniform | 36.67 ± 1.47% | 49.13 | FAIL (0/5) | 3.494 | -136.83 | 77.015 | -69.76 |
| latency | 48 | 2 | learned | 81.86 ± 0.00% | 3.94 | FAIL (0/5) | 0.485 | +43.45 | 14.918 | +43.21 |
| latency | 48 | 2 | matched | 81.58 ± 0.20% | 4.22 | FAIL (0/5) | 0.491 | +42.69 | 15.356 | +41.54 |
| latency | 48 | 2 | uniform | 35.76 ± 2.12% | 50.04 | FAIL (0/5) | 3.494 | -307.57 | 86.202 | -228.17 |
