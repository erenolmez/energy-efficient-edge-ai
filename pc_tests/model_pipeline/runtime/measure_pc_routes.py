"""Matched RGB-input-to-prediction measurements for PC static and routed paths."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import time
import numpy as np
import torch

from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter


def dispatch_index(score, policy):
    if policy['fallback']:
        return int(policy['baseline_index'])
    rank = int(np.clip(np.floor((score + policy['bias']) * len(policy['order'])),
                       0, len(policy['order']) - 1))
    return int(policy['order'][rank])


def measure(images, labels, batches, engines, baseline_index, router=None,
            policy=None, input_kind='features', repeats=5, warmup=20, minimum_seconds=3):
    """Lists contain resident engines; lazy caches load inside timed dispatch.

    Batch indexing and pinning are offline data-loader work, common to both
    paths. Transfer, normalization, GPU features/router, scalar dispatch sync,
    selected TensorRT inference and host predictions are inside the wall timer.
    Neither cached choices nor cached features are accepted by this function.
    """
    if router is not None and policy is None:
        raise ValueError('Routed measurement requires a validation-calibrated policy')
    host_batches = [torch.from_numpy(np.array(images[batch])).pin_memory() for batch in batches]
    if sum(map(len, batches)) != len(labels) or len(np.unique(np.concatenate(batches))) != len(labels):
        raise ValueError('Each held-out image must occur exactly once per sweep')
    inputs = SharedBatchInputs()
    feature_start = torch.cuda.Event(enable_timing=True)
    feature_end = torch.cuda.Event(enable_timing=True)
    model_end = torch.cuda.Event(enable_timing=True)

    def forward(host):
        normalized = inputs.prepare(host)
        feature_start.record()
        if router is None:
            index = baseline_index
        else:
            if input_kind == 'correctness_images':
                from pc_tests.model_pipeline.routers.correctness.correctness_policy import select_correctness
                # Separate exported shapes cover actual partial tails; no padding.
                active_router = router[len(normalized)]
                probabilities = active_router.run(normalized.contiguous()).cpu().numpy().copy()
                if probabilities.shape != (1, len(engines)):
                    raise ValueError('Correctness router must preserve every candidate output')
                index = int(policy['baseline_index']) if policy['fallback'] else int(
                    select_correctness(probabilities, policy['costs'], policy['penalty'])[0])
            elif input_kind == 'image_features':
                values = inputs.image_features(normalized)
                # Fixed-shape TensorRT aggregation engines need a full input.
                # Duplicate the last image's score for partial tails, identically
                # in calibration and test; real image counts are unchanged.
                required = router.shape[0]
                if len(values) < required:
                    values = torch.cat((values, values[-1:].expand(required-len(values), -1)))
            else:
                values = inputs.grid(normalized) if input_kind == 'grid' else inputs.features(normalized)[None, :]
            if input_kind != 'correctness_images':
                score = float(router.run(values.contiguous()).item())
                index = dispatch_index(score, policy)
        feature_end.record()
        output = engines[index].run(normalized, copy_input=False)
        model_end.record()
        predictions = output.argmax(1).cpu().numpy()
        # The host copy synchronizes the dependency chain for these events.
        return predictions, index, feature_start.elapsed_time(feature_end), feature_end.elapsed_time(model_end)

    with torch.inference_mode():
        for iteration in range(warmup):
            forward(host_batches[iteration % len(host_batches)])
        torch.cuda.synchronize()
        measurements = []
        energy = EnergyCounter()
        for repeat in range(repeats):
            # Global cache starts empty each repeat; cold loads remain measured.
            cache_loads = getattr(engines, 'loads', 0)
            cache_evictions = getattr(engines, 'evictions', 0)
            if hasattr(engines, 'clear') and not isinstance(engines, list):
                engines.clear()
            predictions = np.empty(len(labels), dtype=np.int16)
            batch_counts, image_counts = np.zeros(len(engines), dtype=int), np.zeros(len(engines), dtype=int)
            wall_ms, decision_ms, selected_ms = [], [], []
            energy.start()
            energy_start = time.perf_counter()
            measured_images, sweep = 0, 0
            while sweep == 0 or time.perf_counter() - energy_start < minimum_seconds:
                for positions, host in zip(batches, host_batches):
                    started = time.perf_counter()
                    output, index, decision, selected = forward(host)
                    elapsed = (time.perf_counter() - started) * 1000
                    measured_images += len(host)
                    if sweep == 0:
                        predictions[positions] = output
                        batch_counts[index] += 1
                        image_counts[index] += len(host)
                        wall_ms.append(elapsed)
                        decision_ms.append(decision)
                        selected_ms.append(selected)
                sweep += 1
            energy_metrics = energy.finish(measured_images)
            full_test_seconds = float(sum(wall_ms) / 1000)
            measurements.append({'repeat': repeat, 'test_images': len(labels),
                                 'test_accuracy': float(100 * (predictions == labels).mean()),
                                 'rgb_input_to_prediction_seconds': full_test_seconds,
                                 'mean_batch_ms': float(np.mean(wall_ms)),
                                 'p95_batch_ms': float(np.percentile(wall_ms, 95)),
                                 'images_per_second': len(labels) / full_test_seconds,
                                 'mean_gpu_decision_ms': float(np.mean(decision_ms)),
                                 'mean_gpu_selected_inference_ms': float(np.mean(selected_ms)),
                                 'selected_batches': batch_counts.tolist(), 'selected_images': image_counts.tolist(),
                                 'runtime_engine_loads': getattr(engines, 'loads', 0) - cache_loads,
                                 'runtime_engine_evictions': getattr(engines, 'evictions', 0) - cache_evictions,
                                 **energy_metrics})
    return measurements
