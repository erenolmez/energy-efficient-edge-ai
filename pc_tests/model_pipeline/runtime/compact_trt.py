"""Isolated optional host-byte/shared-buffer/graph adapters for future preflight.

Importing this module does not initialize CUDA. Never patch historical runners.
Actual memory/graph parity preflight is required before publishing measurements.
"""
import hashlib
from pathlib import Path


class UnsupportedGraph(RuntimeError):
    pass


def create_runner(record,verified_bytes=None,shared_buffers=None,cuda_graph=False):
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner,TRT_LOGGER,trt

    class CompactRunner(EngineRunner):
        def __init__(self):
            value=verified_bytes if verified_bytes is not None else Path(record['engine_path']).read_bytes()
            # Host store already verified immutable bytes; file mode verifies each load.
            if verified_bytes is None and hashlib.sha256(value).hexdigest()!=record['engine_sha256']:
                raise RuntimeError('Classifier engine hash mismatch')
            self.logger=TRT_LOGGER
            self.runtime=trt.Runtime(self.logger)
            self.engine=self.runtime.deserialize_cuda_engine(value)
            if self.engine is None:
                raise RuntimeError('Engine deserialization failed')
            self.context=self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError('Execution context allocation failed')
            self.stream=torch.cuda.Stream()
            names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_name=next(n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT)
            self.output_name=next(n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT)
            if self.engine.get_tensor_dtype(self.input_name)!=trt.float32:
                raise RuntimeError('Shared FP32 normalized input required')
            if shared_buffers is None:
                self.input=torch.empty((1,3,128,128),device='cuda',dtype=torch.float32)
                self.output=torch.empty((1,100),device='cuda',dtype=torch.float32)
            else:
                self.input,self.output=shared_buffers
                if tuple(self.input.shape)!=(1,3,128,128) or tuple(self.output.shape)!=(1,100):
                    raise ValueError('Shared buffers must be fixed batch1')
            self.context.set_tensor_address(self.input_name,self.input.data_ptr())
            self.context.set_tensor_address(self.output_name,self.output.data_ptr())
            self.graph=None
            if cuda_graph:
                # Capturing one resident context cannot be amortized over evicted contexts.
                try:
                    self.input.zero_()
                    torch.cuda.synchronize()
                    for _ in range(3):
                        super().run(self.input,copy_input=False)
                    torch.cuda.synchronize()
                    self.graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph,stream=self.stream):
                        self._enqueue(self.input,copy_input=False)
                    torch.cuda.synchronize()
                except Exception as error:
                    self.graph=None
                    raise UnsupportedGraph('CUDA Graph capture unsupported; do not silently substitute eager: '+str(error)) from error

        def run(self,images,copy_input=True):
            if self.graph is None:
                return super().run(images,copy_input=copy_input)
            if tuple(images.shape)!=(1,3,128,128) or images.dtype!=torch.float32 or not images.is_cuda:
                raise ValueError('Graph requires fixed batch1 GPU inputs')
            caller=torch.cuda.current_stream()
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                self.input.copy_(images)
                self.graph.replay()
            caller.wait_stream(self.stream)
            return self.output

    return CompactRunner()
