"""Fixed-shape one-input TensorRT runner with typed, named GPU outputs."""
from pathlib import Path

class FixedGPURunner:
    def __init__(self,path):
        import torch
        from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt,TRT_LOGGER
        self.torch=torch;self.runtime=trt.Runtime(TRT_LOGGER)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('Fixedengine deserializationfailed')
        self.context=self.engine.create_execution_context();self.stream=torch.cuda.Stream()
        names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
        if len(inputs)!=1:raise ValueError('Exactly onefixedinput required')
        self.input=inputs[0];self.shape=tuple(self.engine.get_tensor_shape(self.input));self.outputs={}
        types={trt.float32:torch.float32,trt.float16:torch.float16,trt.int32:torch.int32,trt.int64:torch.int64,trt.bool:torch.bool}
        self.input_dtype=types[self.engine.get_tensor_dtype(self.input)]
        if any(d<0 for d in self.shape):raise ValueError('Staticinputshape required')
        for n in names:
            if n==self.input:continue
            shape=tuple(self.engine.get_tensor_shape(n))
            if any(d<0 for d in shape):raise ValueError('Staticoutputshape required')
            self.outputs[n]=torch.empty(shape,device='cuda',dtype=types[self.engine.get_tensor_dtype(n)])
            self.context.set_tensor_address(n,self.outputs[n].data_ptr())
    def run(self,value):
        t=self.torch
        if tuple(value.shape)!=self.shape or value.dtype!=self.input_dtype or not value.is_cuda or not value.is_contiguous():raise ValueError('FixedGPUinput mismatch')
        caller=t.cuda.current_stream();self.stream.wait_stream(caller)
        with t.cuda.stream(self.stream):
            self.context.set_tensor_address(self.input,value.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('FixedGPUexecute failed')
        caller.wait_stream(self.stream)
        return self.outputs
