"""Resumable TensorRT accuracy and matched-runtime PC static benchmarks."""
from __future__ import annotations
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source


import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner, build_engine, trt
from pc_tests.model_pipeline.runtime.engine_precision_audit import integer_compute_layers
from pc_tests.model_pipeline.runtime.prepare_trt_graph import prepare_graph
from pc_tests.model_pipeline.experiments.run_precision_experiments import file_hash, write_json
from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter


def accuracy(runner, images, labels, batch_size=16):
    predictions = []
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(np.array(images[start:start+batch_size])).cuda()
        output = runner.run(x, copy_input=False).cpu().numpy()
        if output.shape != (len(x), 100) or not np.isfinite(output).all():
            raise RuntimeError('Invalid target-runtime output')
        predictions.extend(output.argmax(1).tolist())
    predictions = np.asarray(predictions, dtype=np.int16)
    return 100 * float((predictions == labels).mean()), predictions


def timings(runner, images, batch_size, repeats=5, iterations=30, warmup=20, minimum_seconds=3):
    host_input = torch.from_numpy(np.array(images[:batch_size])).pin_memory()
    device_input = host_input.cuda()
    for _ in range(warmup):
        runner.run(device_input, copy_input=False)
    torch.cuda.synchronize()
    results = []
    energy = EnergyCounter()
    for repeat in range(repeats):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            runner.run(device_input, copy_input=False)
        end_event.record()
        end_event.synchronize()
        forward_ms = start_event.elapsed_time(end_event) / iterations
        energy.start()
        started = time.perf_counter()
        completed = 0
        while completed < iterations or time.perf_counter() - started < minimum_seconds:
            device_input.copy_(host_input, non_blocking=True)
            runner.run(device_input, copy_input=False).cpu()
            completed += 1
        torch.cuda.synchronize()
        host_to_logits_ms = (time.perf_counter() - started) * 1000 / completed
        energy_metrics = energy.finish(completed * batch_size)
        results.append({'repeat': repeat, 'forward_batch_ms': forward_ms,
                        'normalized_host_input_to_logits_batch_ms': host_to_logits_ms,
                        'forward_images_per_second': batch_size * 1000 / forward_ms,
                        'host_input_to_logits_images_per_second': batch_size * 1000 / host_to_logits_ms,
                        'timed_batches': completed, **energy_metrics})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--converted-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--precisions', default='fp32,fp16,int8')
    parser.add_argument('--batch-sizes', default='1,8,16,32,64,128')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    sources = json.loads(args.manifest.read_text())['models']
    if args.models:
        selected = set(args.models.split(','))
        sources = [s for s in sources if s['source_stem'] in selected]
        if len(sources) != len(selected):
            raise ValueError('Unknown source names')
    precisions = args.precisions.split(',')
    if not set(precisions) <= {'fp32','fp16','int8'}:
        raise ValueError('Invalid precision')
    batches = [int(value) for value in args.batch_sizes.split(',')]
    if any(not 1 <= size <= 128 for size in batches):
        raise ValueError('Batch sizes must be between 1 and 128')
    images = np.load(args.cache_dir / 'test.npy', mmap_mode='r')
    labels = np.load(args.cache_dir / 'test_labels.npy')
    validation_images = np.load(args.cache_dir / 'validation.npy', mmap_mode='r')
    validation_labels = np.load(args.cache_dir / 'validation_labels.npy')
    router_data = json.loads((args.cache_dir / 'router_data.json').read_text())
    if len(images) != 10000 or len(labels) != 10000:
        raise ValueError('Complete CIFAR-100 test set required')
    report_path = args.report_dir / 'gpu_static.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'results': []}
    records = {r['name']: r for r in report['results']}
    report.update({'runtime': 'TensorRT GPU', 'tensorrt_version': trt.__version__,
                   'gpu': torch.cuda.get_device_name(0), 'test_images': 10000,
                   'batch_sizes': batches, 'benchmark_repeats': 5,
                   'timing_boundary': 'Forward GPU events, and normalized host tensor transfer to GPU through host logits. Image decoding/resizing/normalization are excluded here and must be included in the later routed pipeline comparisons.',
                   'energy_scope': 'NVML cumulative GPU energy counter measured over >=3-second host-input-to-logits windows; includes GPU idle/display load, excludes CPU and other PC power. Not Jetson VDD_IN.',
                   'validation_indices_sha256': router_data['validation_indices_sha256'],
                   'validation_images': 5000, 'expected': len(sources) * len(precisions)})
    for source in sources:
        for precision in precisions:
            name = f"{source['architecture']}__p{source['prune']}__{precision}"
            destination = args.output_dir / 'engines' / f'{name}.engine'
            graph = Path(source['onnx']) if precision != 'int8' else args.converted_dir / f'{name}.onnx'
            record = {'name': name, 'architecture': source['architecture'],
                      'prune': source['prune'], 'precision': precision}
            started = time.perf_counter()
            try:
                source_hash = file_hash(graph)
                record['source_sha256'] = source_hash
                prior = records.get(name)
                current_engine = (prior and prior.get('source_sha256') == source_hash
                                  and prior.get('tensorrt_version') == trt.__version__
                                  and destination.exists()
                                  and prior.get('engine_sha256') == file_hash(destination))
                if (current_engine and prior.get('status') == 'complete'
                        and prior.get('batch_sizes') == batches and prior.get('protocol_version') == 3
                        and prior.get('validation_indices_sha256') == router_data['validation_indices_sha256']):
                    print(f'Skip {name}', flush=True)
                    continue
                if current_engine:
                    record.update(prior)
                else:
                    if precision == 'int8':
                        prepared = args.output_dir / 'prepared_graphs' / f'{name}.onnx'
                        record['folded_constant_int32_nodes'] = prepare_graph(graph, prepared)
                        graph = prepared
                    print(f'Build {name}', flush=True)
                    build_engine(graph, destination, precision)
                runner = EngineRunner(destination)
                runner.context.set_input_shape(runner.input_name, (8,3,128,128))
                inspector = runner.engine.create_engine_inspector()
                inspector.execution_context = runner.context
                details = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
                integer = integer_compute_layers(details)
                if precision == 'int8' and not integer:
                    raise RuntimeError('No integer convolution/matrix input formats in TensorRT engine')
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / f'{name}_layers.json').write_text(details, encoding='utf-8')
                record.update(engine_sha256=file_hash(destination), engine_bytes=destination.stat().st_size,
                              engine_path=str(destination.resolve()), tensorrt_version=trt.__version__,
                              integer_compute_layers=integer)
                if args.build_only:
                    record['status'] = 'built'
                else:
                    print(f'Evaluate and benchmark {name}', flush=True)
                    value, predictions = accuracy(runner, images, labels)
                    np.save(args.output_dir / f'{name}_test_predictions.npy', predictions)
                    validation_accuracy, validation_predictions = accuracy(runner, validation_images, validation_labels)
                    np.save(args.output_dir / f'{name}_validation_predictions.npy', validation_predictions)
                    record.update(test_accuracy=value, test_images=len(predictions), batch_sizes=batches,
                                  validation_accuracy=validation_accuracy, validation_images=len(validation_predictions),
                                  validation_indices_sha256=router_data['validation_indices_sha256'],
                                  timings={str(size): timings(runner, images, size) for size in batches},
                                  status='complete', protocol_version=3)
                del inspector, runner
            except Exception as exc:
                record.update(status='failed', error=str(exc), traceback=traceback.format_exc())
                print(record['traceback'], flush=True)
            record['stage_seconds'] = time.perf_counter() - started
            records[name] = record
            report['results'] = sorted(records.values(), key=lambda r: r['name'])
            write_json(report_path, report)
            print(f"Finished {name}: {record['status']}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    if any(r['status'] == 'failed' for r in records.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
