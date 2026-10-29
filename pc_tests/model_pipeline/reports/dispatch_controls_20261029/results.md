# Dispatch controls and cold GPU energy

Validation-onlyengineering, nottestaccuracy. GPUonlyenergy excludesCPU. Fixedsameclassifier/context interleaving showedpositiveCNN/readback overhead inall5pairs, but staticlatencyvaried4.10–7.12ms. Selector3repeat rangesoverlap; do notclaimuniversalFP32speedup.15/105near-tiedFP32fixturesdifferedfromFP64 despite0/512realchoicechanges. Aggregatecoldenergy isload+context+cleanupcycles>=2s, notpurestartuporperimage; oldunavailableindividualreadingsnotreplaced.

| Selector | Median ms/image | GPU J/image |
|---|---:|---:|
| cpu64 | 4.352 | 0.2555 |
| torch64 | 4.368 | 0.2505 |
| trt32 | 4.390 | 0.2585 |

| Cold mode | Seconds/load+cleanup | GPU J/load+cleanup |
|---|---:|---:|
| file | 0.439 | 6.584 |
| host | 0.157 | 2.508 |

No verifiedCPU/systemenergy sensor wasavailablethrough thetwo checkedWMIproviders. This doesnotassert that no othermetercouldexist; noCPU/fullPCenergy result orCPUtrainingpaybackreported.
