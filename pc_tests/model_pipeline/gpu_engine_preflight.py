"""Build and execute TensorRT engines on the PC before the full GPU sweep."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Keep DLL handles alive for the process; do not change the system PATH.
DLL_HANDLES = []
CUDA_BIN = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin')
if os.name == 'nt' and CUDA_BIN.exists():
    DLL_HANDLES.append(os.add_dll_directory(str(CUDA_BIN)))
import tensorrt as trt

from run_precision_experiments import file_hash, write_json
from engine_precision_audit import integer_compute_layers
from prepare_trt_graph import prepare_graph

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class EngineRunner:
    """TensorRT execution with reusable CUDA tensors and no CPU operator fallback."""
    def __init__(self, path, maximum_batch=128):
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT deserialization failed')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        if self.engine.get_tensor_dtype(self.input_name) != trt.float32:
            raise ValueError('This runner expects shared FP32 model inputs')
        self.input = torch.empty((maximum_batch, 3, 128, 128), device='cuda', dtype=torch.float32)
        self.output = torch.empty((maximum_batch, 100), device='cuda', dtype=torch.float32)
        self.context.set_tensor_address(self.input_name, self.input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    def run(self, images, copy_input=True):
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            output = self._enqueue(images, copy_input)
        caller.wait_stream(self.stream)
        return output

    def _enqueue(self, images, copy_input=True):
        size = len(images)
        if not images.is_cuda or images.dtype != torch.float32 or not images.is_contiguous():
            raise ValueError('TensorRT input must be a contiguous FP32 CUDA tensor')
        if copy_input:
            self.input[:size].copy_(images)
            address = self.input.data_ptr()
        else:
            # Router and selected model can consume the same preprocessed buffer.
            address = images.data_ptr()
        self.context.set_tensor_address(self.input_name, address)
        if not self.context.set_input_shape(self.input_name, (size, 3, 128, 128)):
            raise RuntimeError('TensorRT rejected dynamic input shape')
        if self.context.get_tensor_shape(self.output_name) != (size, 100):
            raise RuntimeError('Unexpected TensorRT output shape')
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return self.output[:size]


def build_engine(graph, destination, precision, workspace_mib=512):
    logger = TRT_LOGGER
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(graph).read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        # Explicit QDQ encodes calibration scales. FP16 is only a fallback
        # for paths not quantized by the fixed CPU calibration policy.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (1,3,128,128), (32,3,128,128), (128,3,128,128))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.engine.tmp')
    temporary.write_bytes(bytes(serialized))
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--converted-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', default='efficientnet_b0__p0__fp32')
    parser.add_argument('--precisions', default='fp32,fp16,int8')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sources = {s['source_stem']: s for s in manifest['models']}
    images = np.load(args.cache_dir / 'test.npy', mmap_mode='r')
    report_path = args.output_dir / 'preflight.json'
    previous = json.loads(report_path.read_text())['results'] if report_path.exists() else []
    records = {r['name']: r for r in previous}
    for stem in args.models.split(','):
        source = sources[stem]
        for precision in args.precisions.split(','):
            name = f"{source['architecture']}__p{source['prune']}__{precision}"
            graph = Path(source['onnx']) if precision != 'int8' else args.converted_dir / f'{name}.onnx'
            destination = args.output_dir / f'{name}.engine'
            started = time.perf_counter()
            record = {'name': name, 'precision': precision}
            try:
                record['source_sha256'] = file_hash(graph)
                prior = records.get(name)
                if (prior and prior['status'] == 'complete'
                        and prior.get('source_sha256') == record['source_sha256']
                        and destination.exists()
                        and prior.get('engine_sha256') == file_hash(destination)):
                    print(f'Skip completed {name}', flush=True)
                    continue
                if precision == 'int8':
                    compatible_graph = args.output_dir / 'prepared_graphs' / f'{name}.onnx'
                    record['folded_constant_int32_nodes'] = prepare_graph(graph, compatible_graph)
                    record['build_graph_sha256'] = file_hash(compatible_graph)
                    graph = compatible_graph
                print(f'Building {name}', flush=True)
                build_engine(graph, destination, precision)
                runner = EngineRunner(destination)
                runner.context.set_input_shape(runner.input_name, (8,3,128,128))
                inspector = runner.engine.create_engine_inspector()
                inspector.execution_context = runner.context
                details = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
                (args.output_dir / f'{name}_layers.json').write_text(details, encoding='utf-8')
                integer_layers = integer_compute_layers(details)
                record['integer_compute_layers'] = integer_layers
                if precision == 'int8' and not integer_layers:
                    raise RuntimeError('No integer convolution/matrix input formats in engine inspection')
                outputs = []
                for size in (1, 8, 128):
                    x = torch.from_numpy(np.array(images[:size])).cuda()
                    result = runner.run(x).cpu().numpy()
                    if result.shape != (size,100) or not np.isfinite(result).all():
                        raise RuntimeError('TensorRT smoke output invalid')
                    outputs.append({'batch': size, 'finite': True})
                record.update(status='complete', engine_sha256=file_hash(destination),
                              engine_bytes=destination.stat().st_size, smoke=outputs)
                del runner
            except Exception as exc:
                record.update(status='failed', error=str(exc))
            record['seconds'] = time.perf_counter() - started
            records[name] = record
            write_json(report_path, {'tensorrt_version': trt.__version__,
                       'gpu': torch.cuda.get_device_name(0), 'results': list(records.values()),
                       'scope': 'build and execution smoke tests; not latency benchmarks'})
            print(f"{name}: {record['status']}; integer compute layers={len(record.get('integer_compute_layers', []))}; seconds={record['seconds']:.1f}", flush=True)
            torch.cuda.empty_cache()
    if any(r['status'] != 'complete' for r in records.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
