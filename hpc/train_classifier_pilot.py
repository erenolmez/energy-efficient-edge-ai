"""Three-epoch pretrained classifier pilot; never constructs the final test dataset.

This is new-study code, not a continuation of historical PC baselines. It uses an
explicit torchvision checkpoint rather than historical timm defaults. Training
settings are a pilot recipe, not a validation-proven optimum for all models.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import (atomic_write, file_hash, fingerprint,
                                load_or_create_split, validate_resume, write_json)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    return args


def seed_worker(_):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@contextlib.contextmanager
def run_lock(directory):
    """One writer per run; a crash leaves a lock for explicit human inspection."""
    path = directory / "active.lock"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "job": os.environ.get("SLURM_JOB_ID")}, stream)
    try:
        yield
    finally:
        path.unlink()


def main():
    args = arguments()
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from torchvision.datasets import CIFAR100
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import InterpolationMode

    if not args.prepare_only and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training must run in a scheduler allocation")
    if not args.prepare_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing an accidental CPU training run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output_dir):
        # Only train=True: no test object, test labels or router subset loaders.
        raw = CIFAR100(root=args.data_dir, train=True, download=args.download)
        split = load_or_create_split(args.split_manifest, raw.targets, args.split_seed)
        if args.prepare_only:
            print(json.dumps({"split_sha256": split["sha256"],
                              "counts": {key: len(value) for key, value in split["indices"].items()},
                              "test_evaluated": False}), flush=True)
            return
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in
                         [Path(__file__), ROOT / "hpc" / "clean_protocol.py"]}
        recipe = {
            "purpose": "timing/mechanics pilot, not a converged baseline",
            "architecture": "torchvision.resnet18", "weights": "IMAGENET1K_V1",
            "seed": args.seed, "split_seed": args.split_seed, "epochs": args.epochs,
            "batch_size": args.batch_size, "workers": args.workers,
            "image_size": 128, "optimizer": "SGD", "lr": 0.01,
            "momentum": 0.9, "weight_decay": 0.0005, "label_smoothing": 0.1,
            "scheduler": "CosineAnnealingLR", "amp": "float16",
            "normalization": "ImageNet mean/std", "train_resize": 144,
            "train_augmentation": "random crop 128, horizontal flip",
            "validation_transform": "bilinear resize 128, center crop 128",
            "epoch_seed": "seed + epoch; workers recreated each epoch",
            "deterministic_algorithms": True, "tf32": False,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
        }
        checkpoint_path = args.output_dir / "latest.pt"
        if checkpoint_path.exists() and not args.resume:
            raise RuntimeError("Checkpoint exists; use --resume or a fresh output directory")
        if args.resume and not checkpoint_path.exists():
            raise RuntimeError("--resume requested but latest.pt is absent")
        if not checkpoint_path.exists() and (args.output_dir / "run.json").exists():
            raise RuntimeError("Incomplete setup found; inspect before restarting")
        checkpoint = None
        if args.resume:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_resume(checkpoint, recipe, split["sha256"], source_hashes)
        weights = ResNet18_Weights.IMAGENET1K_V1
        started_setup = time.perf_counter()
        initial = None
        if checkpoint is None:
            # Hash and load the exact downloaded state; do not rely on DEFAULT.
            initial = weights.get_state_dict(progress=True, check_hash=True)
            cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
            pretrained = {"identifier": recipe["weights"], "url": weights.url,
                          "file_sha256": file_hash(cached)}
        else:
            pretrained = checkpoint["pretrained"]
        model = resnet18(weights=None)
        if initial is not None:
            model.load_state_dict(initial)
            del initial
        model.fc = torch.nn.Linear(model.fc.in_features, 100)
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=recipe["lr"],
                                    momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda")
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        training = transforms.Compose([
            transforms.Resize(144, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomCrop(128), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize])
        validation = transforms.Compose([
            transforms.Resize(128, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(128), transforms.ToTensor(), normalize])
        training_set = Subset(CIFAR100(args.data_dir, train=True, transform=training),
                              split["indices"]["classifier_train"])
        validation_set = Subset(CIFAR100(args.data_dir, train=True, transform=validation),
                                split["indices"]["classifier_validation"])
        history, first_epoch, best_accuracy, best_state = [], 1, -1.0, None
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            first_epoch = checkpoint["epoch"] + 1
            best_accuracy, best_state = checkpoint["best_accuracy"], checkpoint["best_state"]
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unavailable"
        run = {"recipe": recipe, "recipe_sha256": fingerprint(recipe),
               "split_sha256": split["sha256"], "pretrained": pretrained,
               "source_hashes": source_hashes, "git_revision": revision,
               "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(),
               "job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
               "setup_seconds": time.perf_counter() - started_setup,
               "utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": False}
        # Preserve original run provenance and add separate resume-session metadata.
        metadata_path = args.output_dir / (f"resume_{time.time_ns()}.json" if checkpoint else "run.json")
        write_json(metadata_path, run)
        stop = []
        def request_stop(signum, _frame):
            stop.append(signum)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, request_stop)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, request_stop)
        for epoch in range(first_epoch, args.epochs + 1):
            # Explicit epoch seeds make epoch-boundary resume independent of
            # persistent-worker RNG state; bitwise equality across GPUs is not promised.
            epoch_seed = args.seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            epoch_started = time.perf_counter()
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            for role, dataset in (("train", training_set), ("validation", validation_set)):
                loader = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=role == "train", num_workers=args.workers,
                                    pin_memory=True, persistent_workers=False,
                                    worker_init_fn=seed_worker,
                                    generator=torch.Generator().manual_seed(epoch_seed))
                model.train(role == "train")
                total, correct = 0, torch.zeros((), device="cuda", dtype=torch.int64)
                summed_loss = torch.zeros((), device="cuda")
                torch.cuda.synchronize()
                stage_started = time.perf_counter()
                with torch.set_grad_enabled(role == "train"):
                    for images, labels in loader:
                        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                        if role == "train":
                            optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(images)
                            loss = criterion(logits, labels)
                        if role == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        summed_loss += loss.detach() * labels.numel()
                        correct += (logits.argmax(1) == labels).sum()
                        total += labels.numel()
                torch.cuda.synchronize()
                row[role] = {"images": total, "loss": summed_loss.item() / total,
                             "accuracy": 100 * correct.item() / total,
                             "seconds": time.perf_counter() - stage_started}
                if not math.isfinite(row[role]["loss"]):
                    raise RuntimeError("Non-finite loss; retaining previous checkpoint")
            row["train_and_validation_seconds"] = time.perf_counter() - epoch_started
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            if row["validation"]["accuracy"] > best_accuracy:
                best_accuracy = row["validation"]["accuracy"]
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            scheduler.step()
            history.append(row)
            checkpoint = {"epoch": epoch, "recipe": recipe, "source_hashes": source_hashes,
                          "split_sha256": split["sha256"], "pretrained": pretrained,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                          "history": history, "best_accuracy": best_accuracy,
                          "best_state": best_state}
            checkpoint_started = time.perf_counter()
            atomic_write(checkpoint_path, lambda path: torch.save(checkpoint, path))
            print(json.dumps({**row, "checkpoint_seconds": time.perf_counter() - checkpoint_started,
                              "stop_requested": bool(stop)}), flush=True)
            if stop:
                break
        completed = len(history) == args.epochs
        write_json(args.output_dir / "summary.json", {
            "status": "pilot_complete" if completed else "paused_at_epoch_boundary",
            "epochs_complete": len(history), "target_epochs": args.epochs,
            "history": history, "best_validation_accuracy": best_accuracy,
            "checkpoint_sha256": file_hash(checkpoint_path),
            "split_sha256": split["sha256"], "recipe_sha256": fingerprint(recipe),
            "test_evaluated": False, "energy_measured": False,
            "qualification": "pilot only; no convergence or final accuracy claim"})


if __name__ == "__main__":
    main()
