"""Calibration-ranked hot set and bounded RAM/GPU caches for the batch router."""
from collections import OrderedDict
import hashlib
from pathlib import Path

import torch

from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner, TRT_LOGGER, trt

RESERVE = 1024**3
RAM_LIMIT = 512*1024**2


class BytesRunner(EngineRunner):
    def __init__(self, blob, maximum_batch):
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError('TensorRT deserialization failed')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT context creation failed')
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        if self.engine.get_tensor_dtype(self.input_name) != trt.float32:
            raise ValueError('Expected normalized FP32 input')
        self.input = torch.empty((maximum_batch,3,128,128),device='cuda',dtype=torch.float32)
        self.output = torch.empty((maximum_batch,100),device='cuda',dtype=torch.float32)
        self.context.set_tensor_address(self.input_name,self.input.data_ptr())
        self.context.set_tensor_address(self.output_name,self.output.data_ptr())


class RoutedResidency:
    """Keep a memory-limited prefix of calibration favorites resident."""
    def __init__(self, records, hot, batch, release):
        self.records,self.batch,self.release = records,batch,release
        self.pinned,self.host = {},OrderedDict()
        self.transient = None
        self.transient_id = None
        self.loads = self.evictions = self.disk_reads = self.host_bytes = 0
        self.min_free = torch.cuda.mem_get_info()[0]
        try:
            remaining = {i:self._allowance(r) for i,r in enumerate(records)}
            for i in hot:
                extra = max((v for j,v in remaining.items() if j != i), default=0)
                if torch.cuda.mem_get_info()[0] < RESERVE + extra + remaining[i]:
                    break
                self.pinned[i] = self._load(i)
                if torch.cuda.mem_get_info()[0] < RESERVE + extra:
                    del self.pinned[i]
                    self.release()
                    break
                remaining.pop(i)
            if hot and not self.pinned:
                raise RuntimeError('No ranked classifier fits with reserved space for a transient miss')
        except BaseException:
            self.clear()
            raise

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _allowance(record):
        return 2*record['engine_bytes'] + 64*1024**2

    def _load(self,i):
        record = self.records[i]
        free = torch.cuda.mem_get_info()[0]
        if free < RESERVE + self._allowance(record):
            raise RuntimeError('GPU cache admission would risk the 1 GiB reserve')
        blob = self.host.pop(i,None)
        if blob is None:
            blob = Path(record['engine_path']).read_bytes()
            if hashlib.sha256(blob).hexdigest() != record['engine_sha256']:
                raise ValueError('Classifier hash mismatch: '+record['name'])
            self.disk_reads += 1
            if len(blob) <= RAM_LIMIT:
                while self.host and self.host_bytes+len(blob)>RAM_LIMIT:
                    self.host_bytes -= len(self.host.popitem(last=False)[1])
                self.host[i] = blob
                self.host_bytes += len(blob)
        else:
            self.host[i] = blob
        engine = BytesRunner(blob,self.batch)
        free = torch.cuda.mem_get_info()[0]
        self.min_free = min(self.min_free,free)
        if free < RESERVE:
            raise RuntimeError('GPU free memory crossed the 1 GiB reserve')
        self.loads += 1
        return engine

    def __getitem__(self,i):
        if self.transient_id == i:
            return self.transient
        if self.transient is not None:
            torch.cuda.synchronize()
            self.transient = None
            self.transient_id = None
            self.release()
            self.evictions += 1
        if i in self.pinned:
            return self.pinned[i]
        torch.cuda.synchronize()
        self.transient = self._load(i)
        self.transient_id = i
        return self.transient

    def clear(self):
        torch.cuda.synchronize()
        self.pinned.clear()
        self.transient = None
        self.transient_id = None
        self.host.clear()
        self.host_bytes = 0
        self.release()
