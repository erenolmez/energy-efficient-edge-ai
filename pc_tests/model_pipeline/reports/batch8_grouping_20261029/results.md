# Per-image grouping versus one classifier perbatch8

Actualfulltest, original3seeds3repeats, same6residentclassifiers, truebatch8. FreshperimageGPUgather/groupclassification/scatter/hostoutput charged. Independentmodepolicies calibratedbeforetest; fallbackflags retained. GPU-only energy excludesCPU. Differentfixedbatch1andgroupedclassifiershapes maydiffernumerically. Three-repeat ranges notconfidence intervals; no generalcross-session speedupclaim.

| Seed | Mode | Accuracy % | Pass1pp | ms/image | ms/batch | Classifier calls /10000 | L saving % | GPU-E saving % |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|
| 42 | batch_choice | 83.86 | NO | 0.777 | 6.214 | 1250 | +12.77 | +25.15 |
| 42 | per_image | 84.91 | yes | 2.770 | 22.163 | 2570 | -211.12 | -172.59 |
| 42 | reference | 85.80 | yes | 2.846 | 22.767 | 1250 | -219.60 | -176.12 |
| 42 | eligible_static | 85.85 | yes | 0.890 | 7.124 | 1250 | +0.00 | +0.00 |
| 45 | batch_choice | 83.36 | NO | 0.771 | 6.169 | 1250 | +38.81 | +35.68 |
| 45 | per_image | 84.55 | NO | 3.184 | 25.468 | 2659 | -152.63 | -152.05 |
| 45 | reference | 85.80 | yes | 2.858 | 22.861 | 1250 | -126.77 | -172.60 |
| 45 | eligible_static | 85.85 | yes | 1.260 | 10.081 | 1250 | +0.00 | +0.00 |
| 48 | batch_choice | 84.12 | NO | 0.971 | 7.771 | 1250 | +21.13 | +21.77 |
| 48 | per_image | 84.50 | NO | 3.277 | 26.217 | 2801 | -166.07 | -138.71 |
| 48 | reference | 85.80 | yes | 2.803 | 22.424 | 1250 | -127.57 | -174.69 |
| 48 | eligible_static | 85.85 | yes | 1.232 | 9.854 | 1250 | +0.00 | +0.00 |
