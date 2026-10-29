# Audited correctness-router training energy

New fit-only GPU measurements, not historical energy reconstruction.

| Seed | GPU energy (J) | Fit window (s) | Final BCE |
|---|---:|---:|---:|
| 42 | 270.437 | 11.138 | 0.452966 |
| 45 | 201.185 | 8.389 | 0.449299 |
| 48 | 195.008 | 8.266 | 0.453255 |

Median single-fit energy: 201.185 J; range 195.008-270.437 J.
All three experimental fits together: 666.630 J. Do not charge all three to deployment of one seed.

All 60 epoch checkpoint hashes, historical artifact hashes, source hashes and fit settings verified. Training losses are close but not bitwise identical to the historical runs.

GPU only; gross NVML counter difference including idle/display; not whole-PC.

CUDA context/data validation before measurement; initialization, 20 fit epochs, atomic epoch checkpoint writes inside fit window. Final weights write, calibration, export and supervision generation excluded.

These are three seeds in a single session, not repeated energy trials for each seed. Seed 42 also bears first-fit initialization costs. No precise uncertainty or full-preparation claim is made.

Break-even remains uncomputed: use total joules per deployed fit divided by positive matched end-to-end GPU joules saved per image at acceptable accuracy. Training-presentation energy is not inference energy.
