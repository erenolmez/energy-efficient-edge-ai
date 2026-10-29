"""Selection arithmetic without a probability-vector transfer to the host.

Component only: launching an independent TensorRT context still needs a host ID.
CPU unit tests do not establish GPU speedup or conditional execution support.
"""
import math
import torch
from torch import nn


class DeviceCostSelection(nn.Module):
    def __init__(self,costs,penalty,baseline_index=None):
        super().__init__()
        # Validate immutable costs on CPU once, not by GPU-to-host checks per image.
        values=torch.as_tensor(costs,dtype=torch.float64,device='cpu')
        if values.ndim!=1 or not len(values) or not torch.isfinite(values).all() or (values<=0).any():
            raise ValueError('Finite positive candidate costs required')
        if not math.isfinite(penalty) or penalty<0:
            raise ValueError('Finite nonnegative validation-calibrated penalty required')
        if baseline_index is not None and not 0<=baseline_index<len(values):
            raise ValueError('Unknown fallback candidate')
        self.register_buffer('normalized_costs',values/values.max())
        self.register_buffer('fallback_id',torch.tensor(0 if baseline_index is None else baseline_index,dtype=torch.int64))
        self.penalty=penalty
        self.fallback=baseline_index is not None

    def forward(self,probabilities):
        if probabilities.ndim!=2 or probabilities.shape[1]!=len(self.normalized_costs):
            raise ValueError('Batch-by-candidate probabilities required')
        if probabilities.device!=self.normalized_costs.device:
            raise ValueError('Probabilities and constants must stay on the same device')
        if self.fallback:
            return self.fallback_id.expand(probabilities.shape[0])
        # Float64 mirrors the historical NumPy policy arithmetic. FP32 fusion
        # would be a separate parity-tested ablation, not an invisible change.
        return (probabilities.to(torch.float64)-self.penalty*self.normalized_costs).argmax(1)
