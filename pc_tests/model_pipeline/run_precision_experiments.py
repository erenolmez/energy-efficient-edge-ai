"""Convert and evaluate every supported CIFAR-100 model in FP16 and static INT8."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_pipeline.data import evaluation_transform


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    # Windows readers/scanners can briefly deny replacement. Keep the fully
    # written checkpoint and retry boundedly; permanent errors still propagate.
    for attempt in range(16):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 15:
                raise
            time.sleep(min(.1 * (attempt + 1), .5))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def prepare_data(data_dir, cache_dir, calibration):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, train, indices in (
        ('test', False, list(range(10000))),
        ('calibration', True, calibration['indices']),
    ):
        path = cache_dir / f'{name}.npy'
        label_path = cache_dir / f'{name}_labels.npy'
        expected_shape = (len(indices), 3, 128, 128)
        if path.exists() and label_path.exists():
            existing = np.load(path, mmap_mode='r')
            if existing.shape == expected_shape:
                del existing
                continue
        dataset = CIFAR100(data_dir, train=train, download=False,
                           transform=evaluation_transform(128))
        images = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                          shape=expected_shape)
        labels = []
        for position, index in enumerate(indices):
            image, label = dataset[int(index)]
            images[position] = image.numpy()
            labels.append(label)
        images.flush()
        del images
        np.save(label_path, np.asarray(labels, dtype=np.int64))
    return (np.load(cache_dir / 'test.npy', mmap_mode='r'),
            np.load(cache_dir / 'test_labels.npy'),
            np.load(cache_dir / 'calibration.npy', mmap_mode='r'))


class CalibrationReader(CalibrationDataReader):
    def __init__(self, images, input_name='input', batch_size=16):
        self.images, self.input_name, self.batch_size = images, input_name, batch_size
        self.position = 0

    def get_next(self):
        if self.position >= len(self.images):
            return None
        start = self.position
        self.position += self.batch_size
        return {self.input_name: np.asarray(self.images[start:self.position])}

    def rewind(self):
        self.position = 0


def evaluate_ort(path, images, labels, batch_size):
    runtime = session(path)
    input_name = runtime.get_inputs()[0].name
    correct, count = 0, 0
    predictions = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        output = runtime.run(None, {input_name: np.asarray(images[start:start + batch_size])})[0]
        if output.shape != (len(output), 100) or not np.isfinite(output).all():
            raise RuntimeError('Invalid model output during full test evaluation')
        pred = output.argmax(1)
        predictions.extend(pred.tolist())
        correct += int((pred == labels[start:start + len(pred)]).sum())
        count += len(pred)
        if count % 2048 == 0:
            print(f'  evaluated {count}/{len(images)}', flush=True)
    del runtime
    return {'test_accuracy': 100.0 * correct / count, 'test_images': count,
            'correct': correct, 'evaluation_seconds': time.perf_counter() - started}, predictions


def fp16_convert_evaluate(source, out_dir, images, labels, batch_size):
    model = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
    model.eval().half()
    checkpoint = out_dir / f"{source['architecture']}__p{source['prune']}__fp16.pth"
    graph_path = checkpoint.with_suffix('.onnx')
    if not graph_path.exists() or not checkpoint.exists():
        torch.save(model, checkpoint)
        dummy = torch.zeros(1, 3, 128, 128, dtype=torch.float16)
        with torch.inference_mode():
            torch.onnx.export(model, dummy, str(graph_path), input_names=['input'],
                              output_names=['output'], opset_version=13,
                              dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
        onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path))
    half_weights = sum(t.data_type == onnx.TensorProto.FLOAT16 for t in graph.graph.initializer)
    if half_weights == 0:
        raise RuntimeError('FP16 graph has no FP16 weights')
    del graph
    model.to('cuda')
    correct, count = 0, 0
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(np.array(images[start:start + batch_size])).to('cuda').half()
            output = model(x).float().cpu().numpy()
            if output.shape != (len(x), 100) or not np.isfinite(output).all():
                raise RuntimeError('Invalid FP16 output')
            pred = output.argmax(1)
            predictions.extend(pred.tolist())
            correct += int((pred == labels[start:start + len(pred)]).sum())
            count += len(pred)
    elapsed = time.perf_counter() - started
    # Verify saved ONNX execution at both a single image and a small dynamic batch.
    runtime = session(graph_path)
    for size in (1, 8):
        x = np.array(images[:size]).astype(np.float16)
        result = runtime.run(None, {'input': x})[0]
        if result.shape != (size, 100) or not np.isfinite(result).all():
            raise RuntimeError('FP16 ONNX dynamic-batch check failed')
    del runtime, model
    return {
        'test_accuracy': 100.0 * correct / count, 'correct': correct, 'test_images': count,
        'evaluation_seconds': elapsed, 'runtime': 'PyTorch CUDA FP16',
        'model_path': str(checkpoint.resolve()), 'onnx_path': str(graph_path.resolve()),
        'model_bytes': checkpoint.stat().st_size, 'onnx_bytes': graph_path.stat().st_size,
        'float16_initializers': half_weights, 'onnx_smoke_batches': [1, 8],
    }, predictions


def int8_convert_evaluate(source, out_dir, calibration_images, test_images, labels, batch_size):
    graph_path = out_dir / f"{source['architecture']}__p{source['prune']}__int8.onnx"
    prepared = out_dir / f"{source['architecture']}__p{source['prune']}__preprocessed.onnx"
    if not graph_path.exists():
        quant_pre_process(source['onnx'], str(prepared), skip_optimization=True,
                          skip_symbolic_shape=True)
        print('  calibrating INT8 on 1024 training images', flush=True)
        quantize_static(
            str(prepared), str(graph_path), CalibrationReader(calibration_images),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=['CPUExecutionProvider'],
            extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True},
        )
        onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path))
    int8_weights = sum(t.data_type == onnx.TensorProto.INT8 for t in graph.graph.initializer)
    qdq = sum(node.op_type in ('QuantizeLinear', 'DequantizeLinear') for node in graph.graph.node)
    if int8_weights == 0 or qdq == 0:
        raise RuntimeError('INT8 graph has no quantized weights/QDQ operators')
    node_types = sorted({node.op_type for node in graph.graph.node
                         if node.op_type not in ('QuantizeLinear', 'DequantizeLinear')})
    del graph
    metrics, predictions = evaluate_ort(graph_path, test_images, labels, batch_size)
    # Exercise dynamic shapes not used by the evaluation batch.
    runtime = session(graph_path)
    for size in (1, 8, 128):
        result = runtime.run(None, {'input': np.asarray(test_images[:size])})[0]
        if result.shape != (size, 100) or not np.isfinite(result).all():
            raise RuntimeError('INT8 dynamic-batch check failed')
    del runtime
    return {
        **metrics, 'runtime': 'ONNX Runtime CPU QDQ INT8',
        'onnx_path': str(graph_path.resolve()), 'onnx_bytes': graph_path.stat().st_size,
        'int8_initializers': int8_weights, 'qdq_nodes': qdq,
        'non_qdq_operator_types': node_types, 'onnx_smoke_batches': [1, 8, 128],
        'quantization': 'static MinMax; signed symmetric INT8 activations; per-channel INT8 weights',
        'quantized_operator_types': ['Conv', 'Gemm', 'MatMul'],
        'precision_scope': 'mixed graph: supported convolution/linear paths quantized; remaining operations float',
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--models', default='', help='Optional comma-separated source stems for preflight')
    parser.add_argument('--precisions', default='fp32,fp16,int8')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for FP16 accuracy evaluation')
    sources = json.loads(args.manifest.read_text())['models']
    requested = set(args.models.split(',')) if args.models else None
    if requested:
        sources = [s for s in sources if s['source_stem'] in requested]
        if len(sources) != len(requested):
            raise ValueError('Unknown preflight model')
    precisions = args.precisions.split(',')
    if not set(precisions) <= {'fp32', 'fp16', 'int8'}:
        raise ValueError('Only fp32,fp16,int8 are supported')
    calibration = json.loads(args.calibration_manifest.read_text())
    indices = calibration['indices']
    expected_hash = hashlib.sha256(','.join(str(i) for i in indices).encode('ascii')).hexdigest()
    if calibration['indices_sha256'] != expected_hash or len(indices) != 1024:
        raise ValueError('Invalid calibration indices or count')
    if any(s['split_sha256'] != calibration['training_split_sha256'] for s in sources):
        raise ValueError('Calibration split differs from model training split')
    images, labels, calibration_images = prepare_data(args.data_dir, args.output_dir / 'data_cache', calibration)
    out_models = args.output_dir / 'converted_models'
    out_models.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / 'precision_accuracy.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'results': []}
    report.update({'protocol_version': 1, 'test_dataset': 'CIFAR-100 test, all 10000 images',
                   'image_size': 128, 'calibration_images': 1024,
                   'calibration_indices_sha256': expected_hash,
                   'torch_version': torch.__version__, 'onnxruntime_version': ort.__version__,
                   'gpu': torch.cuda.get_device_name(0), 'expected_results': len(sources) * len(precisions),
                   'scope': 'PC precision conversion and accuracy; no Jetson measurements'})
    baselines = {s['architecture']: s['test_accuracy'] for s in
                 json.loads(args.manifest.read_text())['models'] if s['prune'] == 0}
    results = {r['name']: r for r in report['results']}
    for source in sources:
        for precision in precisions:
            name = f"{source['architecture']}__p{source['prune']}__{precision}"
            previous = results.get(name)
            if (previous and previous['status'] == 'complete'
                    and previous.get('test_images') == 10000
                    and Path(previous['onnx_path']).is_file()
                    and previous.get('source_onnx_sha256') == file_hash(source['onnx'])
                    and previous.get('converted_onnx_sha256') == file_hash(previous['onnx_path'])):
                print(f'Skip completed {name}', flush=True)
                continue
            print(f'Start {name}', flush=True)
            started = time.perf_counter()
            source_hash = file_hash(source['onnx'])
            record = {'name': name, 'architecture': source['architecture'], 'prune': source['prune'],
                      'precision': precision, 'source_onnx_sha256': source_hash,
                      'source_fp32_accuracy': source['test_accuracy'],
                      'p0_fp32_accuracy': baselines[source['architecture']]}
            try:
                if precision == 'fp32':
                    metrics, predictions = evaluate_ort(source['onnx'], images, labels, args.batch_size)
                    metrics.update({'runtime': 'ONNX Runtime CPU FP32',
                                    'onnx_path': source['onnx'],
                                    'onnx_bytes': Path(source['onnx']).stat().st_size})
                elif precision == 'fp16':
                    metrics, predictions = fp16_convert_evaluate(source, out_models, images, labels, args.batch_size)
                else:
                    metrics, predictions = int8_convert_evaluate(source, out_models, calibration_images,
                                                                images, labels, args.batch_size)
                np.save(out_models / f'{name}_predictions.npy', np.asarray(predictions, dtype=np.int16))
                record.update(metrics)
                record.update({'status': 'complete',
                    'precision_loss_pp': source['test_accuracy'] - metrics['test_accuracy'],
                    'loss_vs_p0_fp32_pp': baselines[source['architecture']] - metrics['test_accuracy'],
                    'converted_onnx_sha256': file_hash(record['onnx_path'])})
            except Exception as exc:
                record.update({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
                print(record['traceback'], flush=True)
            record['total_seconds'] = time.perf_counter() - started
            results[name] = record
            report['results'] = sorted(results.values(), key=lambda r: r['name'])
            write_json(report_path, report)
            completed = sum(r['status'] == 'complete' for r in results.values())
            print(f"Finished {name}: {record['status']} ({completed} completed)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    failures = [r for r in results.values() if r['status'] != 'complete']
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
