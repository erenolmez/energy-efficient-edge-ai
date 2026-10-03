#!/usr/bin/env python3
"""Export a batch TinyCNN checkpoint and build its TensorRT FP16 engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 24, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24),
            nn.Conv2d(24, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 48, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 1),
        )

    def forward(self, value):
        return self.net(value).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    return parser.parse_args()


def main():
    args = parse_args()
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyCNN().eval()
    model.load_state_dict(package["state_dict"])
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 3, 32, 32), args.onnx,
        input_names=["batch_grid"], output_names=["difficulty"],
        opset_version=17, do_constant_folding=True,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i))
                                     for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        try:
            config.clear_flag(trt.BuilderFlag.TF32)
        except Exception:
            pass
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT router build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    print(f"Saved {args.engine}")


if __name__ == "__main__":
    main()
