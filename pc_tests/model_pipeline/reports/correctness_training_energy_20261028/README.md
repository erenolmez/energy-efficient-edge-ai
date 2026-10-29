# Correctness-router fit energy, 28 October 2026

Isolated, user-authorized repeat of the historical 213-output correctness fit.
Seeds 42/45/48, 20 epochs, batch 64, Adam lr=.001, unweighted BCEWithLogits,
same 3,000 validation-derived fit positions per seed. No test data are opened.
Original 2,000 calibration positions are not used in this fit-only experiment.
No historical weights, reports, policy calibration or TensorRT engines are replaced.

Runner: `experiments/measure_correctness_training_energy.py`.
Launcher: `scripts/start_correctness_training_energy.ps1`.
Artifacts and atomic `training_energy.json` live in
`artifacts/correctness_training_energy_20261028/`.
Logs and exact executable/script/creation identities are in the same directory.

Gross NVML total-GPU counter energy includes idle/display, not CPU or whole PC.
CUDA context initialization and input audit precede measurement. The measured
window includes model/optimizer initialization, 20 fitting epochs and atomic
epoch checkpoint serialization. Final weights serialization, supervision
generation, calibration and export are excluded. This is newly measured recipe
energy, not reconstructed historical energy and not full preparation energy.
Per-image normalization in the raw meter is per 60,000 training presentations,
NOT deployment inference energy. Use total joules per seed for break-even.

No warm-up training, hyperparameter tuning or test-label selection. Preserve
seed-specific results, hashes and loss differences from original training.
Do not sum three fits as the cost of deploying one seed; report a typical fit
and variation, and separately the total cost of all three experimental fits.
No break-even claim without positive matched inference-energy savings at an
acceptable accuracy, including router/cache overhead. Existing inference
periods have GPU throttling/nonstationarity caveats.

Thermal guard samples every 60 s; two consecutive readings >=85 C latch stop
at the next atomic epoch checkpoint and invoke the previously authorized
graceful shutdown. Worker fails closed on stale/wrong/unreadable/latched state.
CPU temperature is unavailable. Never restart a failed or latched run blindly.
No other experiments, GitHub push, Jetson, E: drive or smart-plug access.
