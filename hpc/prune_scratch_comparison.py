"""Prune, fine-tune, and test one frozen 32px scratch checkpoint."""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

from hpc.clean_protocol import file_hash, validate_split, write_json
from hpc.evaluate_imagenet_comparison import load, require
from hpc.full_factorial_protocol import build_model, checked_plan, trial_directory
from hpc.train_imagenet_comparison import SOURCE_PLAN
from model_pipeline.reproducibility import seed_everything
from pc_tests.model_pipeline.experiments.prune_finetune import make_loader, structurally_prune

# Frozen scratch case: (source seed, fine-tuning batch size).
CASES = {139: (48, 16), 329: (48, 32), 314: (42, 16), 312: (42, 64),
         1678: (48, 32), 237: (48, 32), 228: (45, 64)}


@torch.inference_mode()
def evaluate(model, loader, classes=None):
    model.eval()
    counts = torch.zeros(100, dtype=torch.long, device="cuda")
    hits = torch.zeros_like(counts)
    for x, y in loader:
        y = y.cuda(non_blocking=True)
        logits = model(x.cuda(non_blocking=True))
        require(bool(logits.isfinite().all()), "Non-finite evaluation output")
        counts += torch.bincount(y, minlength=100)
        hits += torch.bincount(y[logits.argmax(1).eq(y)], minlength=100)
    counts, hits = counts.tolist(), hits.tolist()
    accuracy = 100 * sum(hits) / sum(counts)
    if classes is None:
        require(counts == [50] * 100, "Unexpected validation coverage")
        return accuracy
    require(counts == [100] * 100, "Unexpected test coverage")
    scores = dict(zip(classes, hits))
    return dict(test_accuracy=accuracy, per_class_test_accuracy=scores,
                best_classes=[name for name, score in scores.items() if score == max(hits)],
                worst_classes=[name for name, score in scores.items() if score == min(hits)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "data", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("case", type=int, choices=CASES)
    parser.add_argument("level", type=int, choices=range(10, 100, 10))
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available(), "Slurm GPU required")
    seed, batch = CASES[args.case]
    ratio = args.level / 100
    study = args.base / "scratch32_full_factorial_v1"
    out = args.output / f"case_{args.case:04d}"
    plan = checked_plan(study / "plan.json")
    require(plan["sha256"] == SOURCE_PLAN, "Unexpected frozen plan")
    trial = plan["trials"][args.case]
    arch = trial["architecture"]
    src = trial_directory(study, args.case, seed)
    summary = load(src / "summary.json")
    ready = load(study / "data_ready.json")
    split = load(args.base / "split_seed42.json")
    require(trial["seed"] == seed and summary["trial"] == trial
            and summary["status"] == "complete" and summary["epochs_complete"] == 200
            and ready["plan_sha256"] == summary["plan_sha256"] == plan["sha256"]
            and ready["split_sha256"] == summary["split_sha256"] == split["sha256"],
            "Scratch checkpoint metadata mismatch")
    source_hash = file_hash(src / "latest.pt")
    require(source_hash == summary["checkpoint_sha256"], "Scratch checkpoint hash mismatch")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck = torch.load(src / "latest.pt", map_location="cpu", weights_only=True)
    require(ck["epoch"] == 200 and ck["recipe"]["trial"] == trial
            and ck["split_sha256"] == split["sha256"]
            and ck["best_accuracy"] == summary["best_validation_accuracy"]
            and bool(ck["best_state"]), "Best state mismatch")

    out.mkdir(parents=True, exist_ok=False)
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = build_model(arch)
    baseline.load_state_dict(ck["best_state"], strict=True)
    del ck
    model = copy.deepcopy(baseline).cuda().eval()
    pruning = SimpleNamespace(image_size=32, round_to=8, global_pruning=True,
                              isomorphic=True, max_pruning_ratio=.95)
    complexity = structurally_prune(model, ratio, pruning, torch.device("cuda"))
    require(0 < complexity["pruned_parameters"] < complexity["baseline_parameters"],
            "No structural parameter reduction")

    norm = T.Normalize(plan["normalization_mean"], plan["normalization_std"])
    eval_tf = T.Compose([T.ToTensor(), norm])
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    raw = CIFAR100(args.data, train=True, transform=train_tf, download=False)
    validate_split(split, raw.targets)
    require(hashlib.sha256(raw.data.tobytes()).hexdigest() == ready["training_pixels_sha256"],
            "Training data differs from HPC receipt")
    val_data = copy.copy(raw)
    val_data.transform = eval_tf
    train = make_loader(Subset(raw, split["indices"]["classifier_train"]), batch, 2, True, 42)
    val = make_loader(Subset(val_data, split["indices"]["classifier_validation"]), 128, 2, False, 42)
    initial_validation = evaluate(model, val)
    seed_everything(42)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    best = anchor = -1.
    best_epoch = last_significant = 0
    history = []
    best_path = out / "best.pth"

    for epoch in range(1, 26):
        model.train()
        seen = updates = 0
        for x, y in train:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            require(math.isfinite(float(loss.detach())), "Non-finite training loss")
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            updates += scaler.get_scale() >= scale
            seen += len(y)
        require(seen == 40000 and updates > 0, "Unexpected training coverage or all steps skipped")
        acc = evaluate(model, val)
        history.append(dict(epoch=epoch, validation_accuracy=acc))
        if acc > best:
            best, best_epoch = acc, epoch
            pending = out / "best.pending"
            torch.save(copy.deepcopy(model).cpu().eval(), pending)
            pending.replace(best_path)
        if acc >= anchor + .20 - 1e-9:
            anchor, last_significant = acc, epoch
        write_json(out / "progress.json", dict(best_epoch=best_epoch, history=history))
        print(f"{arch} p{args.level}: epoch {epoch}, validation {acc:.2f}%, best {best:.2f}%", flush=True)
        if epoch >= 10 and epoch - last_significant >= 5:
            break

    # Open the holdout only after validation selection has ended.
    del model, opt
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    test_data = CIFAR100(args.data, train=False, transform=eval_tf, download=False)
    test = make_loader(test_data, 128, 2, False, 42)
    before = evaluate(baseline.cuda(), test, test_data.classes)
    baseline.cpu()
    after = evaluate(selected.cuda(), test, test_data.classes)
    write_json(out / "result.json", dict(
        status="complete", case=args.case, architecture=arch, source_seed=seed,
        source_checkpoint_sha256=source_hash, selected_checkpoint_sha256=file_hash(best_path),
        split_sha256=split["sha256"], requested_pruning=ratio, complexity=complexity,
        unpruned_validation=summary["best_validation_accuracy"],
        freshly_pruned_validation=initial_validation, best_validation_accuracy=best,
        best_epoch=best_epoch, epochs_run=epoch, unpruned=before, pruned_finetuned=after))


if __name__ == "__main__":
    main()
