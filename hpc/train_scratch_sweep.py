"""Offline, validation-only scratch CIFAR-100 search; no pretrained downloads.

Prepare once, then run 12 search tasks, rank, and confirm two finalists on two
additional seeds. Test data and router subsets are never instantiated as loaders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from hpc.clean_protocol import atomic_write, file_hash, load_or_create_split, validate_resume, write_json
from hpc.scratch_recipes import (check_confirmation, checked_plan, create_selection,
                                  freeze_plan, trial_directory)
from hpc.train_classifier_pilot import run_lock, seed_worker


def build_model():
    import torch
    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=100)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    torch.nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = torch.nn.Identity()
    return model


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model, trial, plan):
    import torch
    kwargs = {"lr": trial["lr"], "weight_decay": trial["weight_decay"]}
    if trial["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=plan["momentum"],
                               nesterov=plan["nesterov"], **kwargs)
    cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[trial["optimizer"]]
    return cls(model.parameters(), betas=tuple(plan["adam_betas"]), eps=plan["adam_eps"], **kwargs)


def make_scheduler(optimizer, trial, plan):
    import torch
    if trial["schedule"] == "step80_120":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=plan["epochs"])


def cutmix(images, labels, alpha, probability):
    import numpy as np
    import torch
    if alpha <= 0 or np.random.random() >= probability:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    height, width = images.shape[-2:]
    cut_h, cut_w = int(height * math.sqrt(1 - lam)), int(width * math.sqrt(1 - lam))
    center_h, center_w = np.random.randint(height), np.random.randint(width)
    y1, y2 = max(0, center_h - cut_h // 2), min(height, center_h + (cut_h + 1) // 2)
    x1, x2 = max(0, center_w - cut_w // 2), min(width, center_w + (cut_w + 1) // 2)
    order = torch.randperm(images.shape[0], device=images.device)
    images[:, :, y1:y2, x1:x2] = images[order, :, y1:y2, x1:x2]
    actual_lam = 1 - (y2 - y1) * (x2 - x1) / (height * width)
    return images, labels, labels[order], actual_lam


def prepare(args):
    from torchvision.datasets import CIFAR100
    plan = freeze_plan(args.study_dir / "plan.json")
    dataset = CIFAR100(args.data_dir, train=True, download=False)
    split = load_or_create_split(args.split_manifest, dataset.targets, plan["split_seed"])
    receipt = {"plan_sha256": plan["sha256"], "split_sha256": split["sha256"],
               "training_pixels_sha256": hashlib.sha256(dataset.data.tobytes()).hexdigest(),
               "counts": {role: len(ids) for role, ids in split["indices"].items()},
               "test_evaluated": False, "pretrained_weights": False}
    path = args.study_dir / "data_ready.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("Dataset/preparation changed; refusing overwrite")
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)


def train(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires a scheduler allocation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from torchvision.datasets import CIFAR100
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing accidental CPU training")
    plan = checked_plan(args.study_dir / "plan.json")
    ready = json.loads((args.study_dir / "data_ready.json").read_text())
    if args.phase == "search":
        if args.task_id not in range(12):
            raise ValueError("Search task must be 0..11")
        trial_id, seed = args.task_id, plan["search_seed"]
    else:
        if args.task_id not in range(4):
            raise ValueError("Confirmation task must be 0..3")
        selection = json.loads((args.study_dir / "selection.json").read_text())
        trial_id = selection["trial_ids"][args.task_id // 2]
        seed = plan["confirmation_seeds"][args.task_id % 2]
    trial = plan["trials"][trial_id]
    raw = CIFAR100(args.data_dir, train=True, download=False)
    split = load_or_create_split(args.split_manifest, raw.targets, plan["split_seed"])
    if (ready["plan_sha256"] != plan["sha256"] or ready["split_sha256"] != split["sha256"]
            or ready["training_pixels_sha256"] != hashlib.sha256(raw.data.tobytes()).hexdigest()):
        raise ValueError("Dataset does not match preflight receipt")
    check_confirmation(args.study_dir, plan, trial_id, seed, split["sha256"])
    directory = trial_directory(args.study_dir, trial_id, seed)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        recipe = {"plan_sha256": plan["sha256"], "trial": trial, "seed": seed,
                  "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "numpy": np.__version__, "python": sys.version}
        sources = {str(path.relative_to(ROOT)): file_hash(path) for path in (
            Path(__file__), ROOT / "hpc/clean_protocol.py", ROOT / "hpc/scratch_recipes.py",
            ROOT / "hpc/train_classifier_pilot.py")}
        latest = directory / "latest.pt"
        checkpoint = None
        if latest.exists():
            if not args.resume:
                raise ValueError("Existing checkpoint; use --resume")
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            validate_resume(checkpoint, recipe, split["sha256"], sources)
        elif args.resume or (directory / "run.json").exists():
            raise ValueError("Missing checkpoint or incomplete setup; inspect before restarting")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build_model()
        initial_hash = state_fingerprint(model.state_dict())
        model = model.cuda()
        optimizer = make_optimizer(model, trial, plan)
        scheduler = make_scheduler(optimizer, trial, plan)
        scaler = torch.amp.GradScaler("cuda")
        criterion = torch.nn.CrossEntropyLoss()
        history, best_state, best_accuracy, best_loss, first_epoch = [], None, -1.0, float("inf"), 1
        if checkpoint:
            if checkpoint["initial_state_sha256"] != initial_hash:
                raise ValueError("Initialization changed")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history, first_epoch = checkpoint["history"], checkpoint["epoch"] + 1
            best_state, best_accuracy, best_loss = checkpoint["best_state"], checkpoint["best_accuracy"], checkpoint["best_loss"]
        normalize = transforms.Normalize(plan["normalization_mean"], plan["normalization_std"])
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        validation_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_set = Subset(CIFAR100(args.data_dir, train=True, transform=train_transform),
                           split["indices"]["classifier_train"])
        validation_set = Subset(CIFAR100(args.data_dir, train=True, transform=validation_transform),
                                split["indices"]["classifier_validation"])
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unavailable"
        run = {"recipe": recipe, "source_hashes": sources, "split_sha256": split["sha256"],
               "initial_state_sha256": initial_hash, "git_revision": revision,
               "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "host": socket.gethostname(),
               "job": os.environ.get("SLURM_JOB_ID"), "test_evaluated": False,
               "pretrained_weights": False}
        write_json(directory / (f"resume_{time.time_ns()}.json" if checkpoint else "run.json"), run)
        stop = []
        def request_stop(signum, _frame):
            stop.append(signum)
        for signum in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", signal.SIGTERM)):
            signal.signal(signum, request_stop)
        print(json.dumps({"event": "training_ready", "trial_id": trial_id, "seed": seed,
                          "initial_state_sha256": initial_hash, "gpu": run["gpu"]}), flush=True)
        limit = min(plan["epochs"], first_epoch + args.epochs_this_job - 1)
        for epoch in range(first_epoch, limit + 1):
            epoch_seed = seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            torch.cuda.reset_peak_memory_stats()
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            for role, dataset in (("train", train_set), ("validation", validation_set)):
                loader = DataLoader(dataset, batch_size=plan["batch_size"], shuffle=role == "train",
                    num_workers=plan["workers"], pin_memory=True, persistent_workers=False,
                    worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(epoch_seed))
                model.train(role == "train")
                correct = torch.zeros((), device="cuda")
                loss_sum = torch.zeros((), device="cuda")
                total = 0
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.set_grad_enabled(role == "train"):
                    for batch_id, (images, labels) in enumerate(loader):
                        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                        targets_a, targets_b, lam = labels, labels, 1.0
                        if role == "train":
                            images, targets_a, targets_b, lam = cutmix(images, labels, trial["cutmix_alpha"], plan["cutmix_probability"])
                            optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(images)
                            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                        if role == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        loss_sum += loss.detach() * labels.numel()
                        predicted = logits.argmax(1)
                        correct += lam * (predicted == targets_a).sum() + (1 - lam) * (predicted == targets_b).sum()
                        total += labels.numel()
                        if role == "train" and batch_id % 100 == 0:
                            print(json.dumps({"event": "batch", "epoch": epoch, "images": total,
                                              "trial_id": trial_id, "seed": seed}), flush=True)
                torch.cuda.synchronize()
                row[role] = {"images": total, "loss": loss_sum.item() / total,
                             "accuracy": 100 * correct.item() / total,
                             "seconds": time.perf_counter() - started}
                if not math.isfinite(row[role]["loss"]):
                    raise RuntimeError("Non-finite loss; previous checkpoint remains intact")
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            score = row["validation"]
            if score["accuracy"] > best_accuracy or (score["accuracy"] == best_accuracy and score["loss"] < best_loss):
                best_accuracy, best_loss = score["accuracy"], score["loss"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            scheduler.step()
            history.append(row)
            checkpoint = {"epoch": epoch, "recipe": recipe, "source_hashes": sources,
                "split_sha256": split["sha256"], "initial_state_sha256": initial_hash,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "history": history, "best_state": best_state,
                "best_accuracy": best_accuracy, "best_loss": best_loss}
            atomic_write(latest, lambda temporary: torch.save(checkpoint, temporary))
            print(json.dumps({"event": "epoch_saved", "trial_id": trial_id, "seed": seed, **row}), flush=True)
            if stop:
                break
        result = {"status": "complete" if len(history) == plan["epochs"] else "paused",
                  "epochs_complete": len(history), "plan_sha256": plan["sha256"],
                  "trial": trial, "seed": seed, "split_sha256": split["sha256"],
                  "initial_state_sha256": initial_hash, "source_hashes": sources,
                  "checkpoint_sha256": file_hash(latest), "best_validation_accuracy": best_accuracy,
                  "best_validation_loss": best_loss, "history": history,
                  "training_seconds": sum(row["train"]["seconds"] + row["validation"]["seconds"] for row in history),
                  "test_evaluated": False, "energy_measured": False,
                  "training_accuracy_note": "CutMix runs report label-weighted training accuracy"}
        write_json(directory / "summary.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "history"}), flush=True)


def rank(args):
    plan = checked_plan(args.study_dir / "plan.json")
    rows, selection = create_selection(args.study_dir, plan)
    with (args.study_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trial_id", "accuracy", "loss", "training_seconds", "recipe"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"leaderboard": rows, "confirmation": selection}), flush=True)


def finalize(args):
    plan = checked_plan(args.study_dir / "plan.json")
    selection = json.loads((args.study_dir / "selection.json").read_text())
    rows = []
    for trial_id in selection["trial_ids"]:
        results = []
        for seed in plan["confirmation_seeds"]:
            path = trial_directory(args.study_dir, trial_id, seed) / "summary.json"
            result = json.loads(path.read_text())
            check_confirmation(args.study_dir, plan, trial_id, seed, result["split_sha256"])
            if (result["status"] != "complete" or result["plan_sha256"] != plan["sha256"]
                    or result["epochs_complete"] != 200 or result["trial"] != plan["trials"][trial_id]
                    or result["seed"] != seed or result["test_evaluated"]):
                raise ValueError("Incomplete/incompatible confirmation")
            results.append({"seed": seed, "accuracy": result["best_validation_accuracy"],
                            "loss": result["best_validation_loss"]})
        rows.append({"trial_id": trial_id, "accuracy": sum(x["accuracy"] for x in results) / 2,
                     "loss": sum(x["loss"] for x in results) / 2, "confirmation_results": results})
    from hpc.scratch_recipes import rank_rows
    rows = rank_rows(rows)
    write_json(args.study_dir / "winner.json", {"plan_sha256": plan["sha256"],
               "winner_trial_id": rows[0]["trial_id"], "ranked_confirmation": rows,
               "selection_basis": "mean validation across seeds45/48, not best seed",
               "test_evaluated": False, "qualification": "selected within this bounded ResNet18 search only"})
    print(json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "rank", "finalize"))
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--phase", choices=("search", "confirm"), default="search")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs-this-job", type=int, default=200)
    args = parser.parse_args()
    if args.action in ("prepare", "train") and (args.data_dir is None or args.split_manifest is None):
        parser.error("prepare/train require --data-dir and --split-manifest")
    if args.epochs_this_job < 1:
        parser.error("--epochs-this-job must be positive")
    {"prepare": prepare, "train": train, "rank": rank, "finalize": finalize}[args.action](args)


if __name__ == "__main__":
    main()
