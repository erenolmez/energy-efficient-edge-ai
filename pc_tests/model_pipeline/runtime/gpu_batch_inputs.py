"""Shared CUDA input buffers, normalization and low-cost batch features."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import torch
import torch.nn.functional as F


class SharedBatchInputs:
    def __init__(self, maximum_batch=128, device='cuda'):
        self.device = device
        self.rgb = torch.empty(maximum_batch,3,128,128,dtype=torch.uint8,device=device)
        self.normalized = torch.empty(maximum_batch,3,128,128,dtype=torch.float32,device=device)
        self.mean = torch.tensor([.5071,.4867,.4408],device=device)[None,:,None,None]
        self.std = torch.tensor([.2675,.2565,.2761],device=device)[None,:,None,None]

    def prepare(self, host_rgb):
        count = len(host_rgb)
        if host_rgb.dtype != torch.uint8 or host_rgb.shape[1:] != (3,128,128):
            raise ValueError('Expected NCHW RGB128 uint8 input')
        if count > len(self.rgb):
            raise ValueError('Batch exceeds allocated shared buffers')
        self.rgb[:count].copy_(host_rgb, non_blocking=True)
        self.normalized[:count].copy_(self.rgb[:count])
        self.normalized[:count].div_(255).sub_(self.mean).div_(self.std)
        return self.normalized[:count]

    def features(self, normalized):
        # Work on a 16x16 thumbnail and retain whole-batch statistics. Features
        # are constructed on GPU during each decision, never loaded from cache.
        rgb = F.adaptive_avg_pool2d(normalized * self.std + self.mean, (16,16))
        gray = rgb[:,0]*.2989 + rgb[:,1]*.5870 + rgb[:,2]*.1140
        dx = (gray[:,:,1:]-gray[:,:,:-1]).abs().flatten(1)
        dy = (gray[:,1:,:]-gray[:,:-1,:]).abs().flatten(1)
        gradient = torch.cat((dx,dy),dim=1)
        saturation = rgb.amax(1)-rgb.amin(1)
        per_image_brightness = gray.mean((1,2))
        return torch.cat((rgb.mean((0,2,3)), rgb.std((0,2,3),unbiased=False), torch.stack((
            gray.mean(), gray.std(unbiased=False), gradient.mean(),
            gradient.std(unbiased=False), (gradient>.1).float().mean(),
            saturation.mean(), per_image_brightness.std(unbiased=False)))))

    def grid(self, normalized):
        selected = torch.linspace(0,len(normalized)-1,16,device=normalized.device).round().long()
        tiles = F.adaptive_avg_pool2d(normalized[selected],(8,8))
        return tiles.reshape(4,4,3,8,8).permute(2,0,3,1,4).reshape(1,3,32,32).contiguous()

    def image_features(self, normalized):
        """Per-image counterpart for the percentile/mean aggregation controls."""
        rgb = F.adaptive_avg_pool2d(normalized * self.std + self.mean, (16,16))
        gray = rgb[:,0]*.2989 + rgb[:,1]*.5870 + rgb[:,2]*.1140
        gradient = torch.cat(((gray[:,:,1:]-gray[:,:,:-1]).abs().flatten(1),
                              (gray[:,1:,:]-gray[:,:-1,:]).abs().flatten(1)), 1)
        return torch.cat((rgb.mean((2,3)), rgb.std((2,3),unbiased=False),
                          torch.stack((gray.mean((1,2)), gray.std((1,2),unbiased=False),
                                       gradient.mean(1), gradient.std(1,unbiased=False),
                                       (gradient>.1).float().mean(1),
                                       (rgb.amax(1)-rgb.amin(1)).mean((1,2)),
                                       torch.zeros(len(rgb),device=rgb.device)), 1)), 1)
