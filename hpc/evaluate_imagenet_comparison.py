"""Evaluate the frozen ImageNet comparison's best checkpoints on CIFAR-100 test."""
import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

from hpc.clean_protocol import file_hash, fingerprint, write_json
from hpc.full_factorial_protocol import checked_plan
from hpc.train_imagenet_comparison import CASES, SOURCE_PLAN, STUDY
from model_pipeline.models import create_classifier


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "data", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    require(torch.cuda.is_available(), "CUDA allocation required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scratch = checked_plan(args.base / "scratch32_full_factorial_v1" / "plan.json")
    study = args.base / STUDY
    prepared = load(study / "prepared.json")
    spec = prepared["spec"]
    require(scratch["sha256"] == spec["scratch_plan_sha256"] == SOURCE_PLAN, "Plan mismatch")
    require(spec["cases"] == list(CASES) and spec["sizes"] == [32, 128], "Unexpected cases")
    registry = "model_pipeline/models.py"
    require(file_hash(Path(registry)) == spec["source_hashes"][registry], "Registry mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for size in (32, 128):
        resize = [] if size == 32 else [T.Resize((size, size), antialias=True)]
        transform = T.Compose([*resize, T.ToTensor(),
                               T.Normalize(scratch["normalization_mean"], scratch["normalization_std"])])
        test = CIFAR100(args.data, train=False, transform=transform, download=False)
        require(len(test) == 10000 and all(test.targets.count(i) == 100 for i in range(100)),
                "Unexpected test set")
        loader = torch.utils.data.DataLoader(test, batch_size=128, num_workers=2,
                                              pin_memory=True, persistent_workers=True)
        for cid in CASES:
            trial = scratch["trials"][cid]
            directory = study / f"case_{cid:04d}_{size}px"
            summary = load(directory / "summary.json")
            recipe = summary["recipe"]
            require(summary["status"] == "complete" and summary["epochs_complete"] == 200
                    and recipe["trial"] == trial and recipe["image_size"] == size
                    and recipe["plan_sha256"] == fingerprint(prepared)
                    and recipe["initial"] == prepared["weights"][str(cid)],
                    f"Summary mismatch: {directory.name}")
            path = directory / "latest.pt"
            digest = file_hash(path)
            require(digest == summary["checkpoint_sha256"], f"Hash mismatch: {path}")
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                ck = torch.load(path, map_location="cpu", weights_only=True)
            require(ck["epoch"] == 200 and ck["recipe"] == recipe
                    and ck["source_hashes"] == spec["source_hashes"]
                    and ck["split_sha256"] == spec["split_sha256"]
                    and ck["best_accuracy"] == summary["best_validation_accuracy"]
                    and bool(ck["best_state"]), f"Best checkpoint mismatch: {path}")
            model = create_classifier(trial["architecture"], pretrained=False)
            model.load_state_dict(ck["best_state"], strict=True)
            del ck
            model = model.cuda().eval()
            correct = seen = 0
            with torch.inference_mode():
                for x, y in loader:
                    logits = model(x.cuda(non_blocking=True))
                    require(bool(logits.isfinite().all()), "Non-finite test logits")
                    correct += int(logits.argmax(1).cpu().eq(y).sum())
                    seen += len(y)
            require(seen == 10000, "Incomplete test evaluation")
            row = dict(architecture=trial["architecture"], case=cid, seed=trial["seed"],
                       size=size, best_validation_accuracy=summary["best_validation_accuracy"],
                       test_accuracy=correct / 100, test_images=seen, checkpoint_sha256=digest)
            rows.append(row)
            write_json(args.output / "partial.json", rows)
            print(json.dumps(row), flush=True)
            del model
        del loader
    require(len(rows) == 16, "Expected all 16 results")
    write_json(args.output / "results.json", dict(precision="PyTorch FP32", rows=rows))


if __name__ == "__main__":
    main()
