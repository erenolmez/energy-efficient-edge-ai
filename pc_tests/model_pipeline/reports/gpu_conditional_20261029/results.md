# Native GPU-conditional two-classifier prototype

TwoFP32classifiers/sameoldseed42CNN, actualnativeIfGPUconditionandtypedbranchoutput, onlyselectedclassifiercomputes.512validationimages,3repeats. No intermediatebranchIDreadback intimedGPUmode; finalCPUclassmatched.

Engineeringmedianbias fromcalibrationdeliberatelyexercisesbothbranches; NOT validated1ppdeploymentpolicy orheldouttestaccuracy. Both-eager isnegativecontrol, notstandardrouterbaseline. GenuineGPUcontrol neednotbefasterthanCPUdispatch. Runtimeprofilerremovedbynewcontextbeforetiming. GPU-onlyenergy excludesCPU. No general213modelorallarchitectureclaim.

| Implementation | Median ms/image | GPU J/image |
|---|---:|---:|
| cpu_dispatch | 3.955 | 0.2286 |
| gpu_conditional | 4.123 | 0.2714 |
| both_eager | 6.596 | 0.4340 |

Conditional vs CPU dispatch latency saving: -4.24%. Conditional vs both-eager negative control: +37.50%.

Chosenbranchlogit maximum absolute difference 0.00001788; zero class/choice mismatches on512validationimages. Bothbranchestested; unselectedConv/Gemm profilemass zero within1e-7ms threshold.
