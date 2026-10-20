"""Build and run the complete learned batch decision in TensorRT."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

from pathlib import Path
import numpy as np
import torch

from pc_tests.model_pipeline.runtime.gpu_engine_preflight import TRT_LOGGER, trt


class RouterRunner:
    def __init__(self, path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Router engine deserialization failed')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        self.shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if self.engine.get_tensor_dtype(self.input_name) != trt.float32:
            raise ValueError('Router requires FP32 shared GPU features or grid')
        self.output = torch.empty(tuple(self.engine.get_tensor_shape(self.output_name)), device='cuda')
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    def run(self, values):
        if tuple(values.shape) != self.shape or not values.is_cuda or not values.is_contiguous() or values.dtype != torch.float32:
            raise ValueError('Invalid router GPU input')
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            self.context.set_tensor_address(self.input_name, values.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('TensorRT router execution failed')
        caller.wait_stream(self.stream)
        return self.output


def build_router(model, input_kind, directory, name, input_shape=None):
    """Export locally trained weights, then check engine/torch numerical parity."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = input_shape or ((1, 3, 32, 32) if input_kind == 'grid' else (1, 13))
    example = torch.zeros(shape, device='cuda')
    graph, path = directory / f'{name}.onnx', directory / f'{name}.engine'
    model = model.cuda().eval()
    torch.onnx.export(model, example, graph, input_names=['input'], output_names=['difficulty'],
                      opset_version=17, dynamo=False)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(graph.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 * 1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT router build failed')
    temporary = path.with_suffix('.engine.tmp')
    temporary.write_bytes(bytes(serialized))
    temporary.replace(path)
    runner = RouterRunner(path)
    generator = torch.Generator(device='cuda').manual_seed(42)
    maximum_error = 0.
    with torch.inference_mode():
        for _ in range(100):
            inputs = torch.randn(shape, generator=generator, device='cuda')
            expected = model(inputs).cpu().numpy()
            actual = runner.run(inputs).cpu().numpy()
            if not np.isfinite(actual).all():
                raise RuntimeError('Router produced nonfinite TensorRT outputs')
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            if not np.allclose(expected, actual, atol=1e-5, rtol=1e-4):
                raise RuntimeError(f'Router numerical parity failed: {maximum_error}')
    return path, maximum_error
