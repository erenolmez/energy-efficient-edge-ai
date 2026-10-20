"""Fast construction and forward-pass check for every selected architecture."""
from __future__ import annotations
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source


import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = REPOSITORY_ROOT
sys.path.insert(0, str(REPOSITORY_ROOT))

from model_pipeline.models import ARCHITECTURES, create_classifier  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    sample = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    rows = []
    for architecture in ARCHITECTURES:
        model = create_classifier(architecture).to(device).eval()
        with torch.inference_mode():
            output = model(sample)
        row = {
            "architecture": architecture,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(output.shape),
            "status": "pass" if tuple(output.shape) == (1, 100) else "fail",
        }
        rows.append(row)
        print(json.dumps(row))
        del model
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
