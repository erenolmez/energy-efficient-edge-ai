# FP32 input-only router benchmark

This is the active CIFAR-100 experiment. It evaluates 22 input-only routing
methods against the same unpruned `fp32_p0` baseline. Energy and latency are
always separate experiments; the code does not combine them into one score.

## Protocol

- Candidate pool: distilled FP32 ResNet-18 at p0, p10, ..., p90 pruning.
- Split: fixed class-stratified 6,000 fit / 2,000 calibration / 2,000 evaluation.
- Learner seeds: 42 through 51 by default.
- Runtime input: only the image, represented as handcrafted features, frozen
  MobileNet embeddings, or direct thumbnail/full-resolution pixels.
- Calibration: minimize selected-candidate cost while remaining within one
  percentage point of p0 accuracy.
- Reporting baseline: `fp32_p0`, never the offline cheapest-correct upper bound.

The 22 methods cover basic, enhanced, compact, HOG/LBP texture, feature-selected
and regularized XGBoost; Extra Trees; Random Forest; Ridge; k-NN; frozen
MobileNetV3-Small embeddings; TinyCNN at 8x8, 16x16, and 32x32; and a thumbnail
MLP. Scalar and per-model correctness/advantage targets are both included.

## Environment

From the repository root:

```powershell
C:\Users\Eren\radioconda\python.exe -m venv .cuda-venv
.\.cuda-venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r pc_tests\router_benchmark\requirements.txt
```

PyTorch neural-router training and frozen-embedding extraction use CUDA.
XGBoost and other scikit-learn routers use CPU inference. PC router latency and
historical Jetson candidate costs are reported separately.

## Rebuild supervision inputs

These commands are needed only when the candidate catalogue, checkpoints, or
Jetson benchmark changes:

```powershell
cd pc_tests\router_benchmark
Copy-Item models.fp32.csv artifacts\models_fp32.csv -Force

python extract_cifar100_features.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output artifacts\features.csv

python collect_pytorch_predictions.py `
  --models artifacts\models_fp32.csv `
  --checkpoint-root "C:\Users\Eren\Desktop\tez\pruning + quantization" `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output artifacts\predictions_all_fp32.csv `
  --batch-size 64 --splits test

python build_model_costs.py `
  --benchmark "C:\Users\Eren\Desktop\tez\tensorrt_results_cifar100_distilled\tensorrt_summary_by_model.csv" `
  --models artifacts\models_fp32.csv `
  --latency-column latency_e2e_mean `
  --objective energy `
  --output artifacts\costs_energy_all_fp32.csv

```

Repeat the cost command with `--objective latency` and the latency filename.
The runner reads neutral image features and labels from `features.csv`, then
constructs objective-specific supervision directly from cached predictions and
the selected energy or latency cost table.

## Run the current benchmark

Run the objectives independently:

```powershell
python run_refined_routers.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output-dir artifacts\refined_energy `
  --objective energy

python run_refined_routers.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output-dir artifacts\refined_latency `
  --objective latency
```

Use `--seeds`, `--methods`, `--epochs`, or `--latency-samples` to override the
documented defaults. The runner saves after every router/seed and resumes
completed work.

Use `--candidate-models` when the deployment benchmark contains only a subset
of pruning levels. Training and policy calibration then use exactly that same
set; for example:

```powershell
python run_refined_routers.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output-dir artifacts\tinycnn4_energy `
  --objective energy `
  --methods tinycnn8_scalar `
  --candidate-models fp32_p0,fp32_p30,fp32_p50,fp32_p70
```

If seeds were produced in multiple directories, merge one objective with:

```powershell
python combine_seed_runs.py `
  --objective energy `
  --sources artifacts\batch_42_44 artifacts\batch_45_51 `
  --output artifacts\energy_10seeds `
  --report ENERGY_RESULTS.md
```

## Batch-level routers

`train_batch_routers.py` trains one-decision-per-batch routers for the
`p0/p10/p20/p30` candidate set. It creates a gradient stump, depth-3 tree,
logistic model, shallow XGBoost model, and three direct TinyCNN seeds. Policies
are calibrated separately for energy and latency at 0.5, 1.0, 1.5, and 2.0
percentage-point accuracy budgets.

```powershell
python train_batch_routers.py `
  --data-dir "C:\path\to\cifar100" `
  --output-dir artifacts\batch_routers `
  --batch-sizes 32,64,128 `
  --budgets-pp 0.5,1.0,1.5,2.0 `
  --seeds 45,48,51 `
  --static-costs-energy artifacts\static_energy.csv `
  --static-costs-latency artifacts\static_latency.csv
```

The optional static-cost files let policy calibration use the measured Jetson
cost of each candidate at each batch size.

## Current reports

- `ENERGY_RESULTS.md`: separate 10-seed candidate-energy experiment.
- `LATENCY_RESULTS.md`: separate 10-seed candidate-latency experiment.
- Each report includes every router, p0-relative accuracy, pass count, router
  overhead, and average model-selection counts for the recommended methods.

Candidate energy excludes router energy. Candidate latency excludes router
latency. Final deployment claims require measuring the entire router plus
selected-model pipeline on the Jetson.
