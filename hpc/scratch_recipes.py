"""Versioned, bounded search; no external data, account or tracking service."""
from __future__ import annotations

import json
import math
from pathlib import Path

from hpc.clean_protocol import file_hash, fingerprint, write_json


def make_plan():
    trials = []
    for optimizer, rates, decay in (
        ("sgd", (0.03, 0.1, 0.2), 5e-4),
        ("adam", (3e-4, 1e-3, 3e-3), 1e-4),
        ("adamw", (3e-4, 1e-3, 3e-3), 0.01),
    ):
        for rate in rates:
            trials.append({"optimizer": optimizer, "lr": rate, "weight_decay": decay,
                           "schedule": "cosine", "cutmix_alpha": 0.0})
    # Published residual-network reference: ICLR 2025, Appendix H.
    trials.append({"optimizer": "sgd", "lr": 0.1, "weight_decay": 1e-4,
                   "schedule": "step80_120", "cutmix_alpha": 0.0})
    # Paired recipe ablations, not a huge optimizer x augmentation Cartesian grid.
    for optimizer, rate, decay in (("sgd", 0.1, 5e-4), ("adamw", 1e-3, 0.01)):
        trials.append({"optimizer": optimizer, "lr": rate, "weight_decay": decay,
                       "schedule": "cosine", "cutmix_alpha": 1.0})
    for number, trial in enumerate(trials):
        trial["id"] = number
    plan = {
        "schema": 1, "study": "scratch32_resnet18_v1",
        "architecture": "torchvision_resnet18_cifar_stem_v1",
        "stem": "3x3 stride1 padding1 convolution, no initial maxpool",
        "weights": None, "classes": 100, "image_size": 32,
        "epochs": 200, "batch_size": 128, "workers": 4,
        "momentum": 0.9, "nesterov": False, "adam_betas": [0.9, 0.999],
        "adam_eps": 1e-8, "label_smoothing": 0.0,
        "normalization_mean": [0.5071, 0.4867, 0.4408],
        "normalization_std": [0.2675, 0.2565, 0.2761],
        "augmentation": "RandomCrop(32,padding=4),RandomHorizontalFlip",
        "cutmix_probability": 0.5, "cutmix_label_weight": "actual clipped patch area",
        "amp": "float16", "deterministic_algorithms": True, "tf32": False,
        "split_seed": 42, "search_seed": 42, "confirmation_seeds": [45, 48],
        "confirmation_finalists": 2, "trials": trials,
        "selection": "best classifier-validation accuracy; loss then trial ID break ties",
        "final_selection": "mean confirmation-seed best validation accuracy; loss then ID ties",
        "data_roles": {"classifier_train": 40000, "classifier_validation": 5000,
                       "router_train": 3000, "router_calibration": 2000},
        "test_policy": "official test never instantiated; no test-based ranking",
        "maximum_training_epochs": 12 * 200 + 2 * 2 * 200,
        "qualification": "literature-informed candidate recipes, not known optima; only ResNet18",
        "sources": [
            "https://proceedings.iclr.cc/paper_files/paper/2025/file/1eac91b56dc20d4f2046951453dde527-Paper-Conference.pdf",
            "https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_cifar.md",
            "https://openreview.net/pdf?id=fWNHKHh0Yn",
        ],
    }
    plan["sha256"] = fingerprint(plan)
    return plan


def freeze_plan(path):
    path = Path(path)
    expected = make_plan()
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("Search plan changed; use a new study directory")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    # The coordinator creates this BEFORE any training array starts.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(expected, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return expected


def checked_plan(path):
    actual = json.loads(Path(path).read_text(encoding="utf-8"))
    if actual != make_plan():
        raise ValueError("Unrecognized or modified frozen search plan")
    return actual


def trial_directory(root, trial_id, seed):
    if trial_id not in range(12) or seed not in (42, 45, 48):
        raise ValueError("Unknown trial or seed")
    return Path(root) / f"trial_{trial_id:02d}" / f"seed_{seed}"


def rank_rows(rows):
    for row in rows:
        if not all(math.isfinite(row[field]) for field in ("accuracy", "loss")):
            raise ValueError("Non-finite metric")
    return sorted(rows, key=lambda row: (-row["accuracy"], row["loss"], row["trial_id"]))


def ranked_search(root, plan):
    rows = []
    split_hash = None
    initialization_hash = None
    reference_sources = None
    for trial in plan["trials"]:
        directory = trial_directory(root, trial["id"], plan["search_seed"])
        result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if (result["status"] != "complete" or result["epochs_complete"] != plan["epochs"]
                or result["plan_sha256"] != plan["sha256"] or result["trial"] != trial
                or result["seed"] != plan["search_seed"] or result["test_evaluated"]):
            raise ValueError("Cannot rank an incomplete or incompatible trial")
        if split_hash is not None and split_hash != result["split_sha256"]:
            raise ValueError("Trials used different dataset splits")
        split_hash = result["split_sha256"]
        if reference_sources is not None and reference_sources != result["source_hashes"]:
            raise ValueError("Search trials used different measured source code")
        reference_sources = result["source_hashes"]
        if initialization_hash is not None and initialization_hash != result["initial_state_sha256"]:
            raise ValueError("Trials did not start from identical weights")
        initialization_hash = result["initial_state_sha256"]
        if file_hash(directory / "latest.pt") != result["checkpoint_sha256"]:
            raise ValueError("Saved checkpoint hash mismatch")
        rows.append({"trial_id": trial["id"], "accuracy": result["best_validation_accuracy"],
                     "loss": result["best_validation_loss"],
                     "training_seconds": result["training_seconds"], "recipe": trial})
    return rank_rows(rows), split_hash


def create_selection(root, plan):
    rows, split_hash = ranked_search(root, plan)
    selection = {"plan_sha256": plan["sha256"], "split_sha256": split_hash,
                 "trial_ids": [row["trial_id"] for row in rows[:2]],
                 "confirmation_seeds": plan["confirmation_seeds"],
                 "test_evaluated": False}
    selection["sha256"] = fingerprint(selection)
    path = Path(root) / "selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Selection changed after freezing; preserve existing study")
    if not path.exists():
        write_json(path, selection)
    return rows, selection


def check_confirmation(root, plan, trial_id, seed, split_hash):
    if seed == plan["search_seed"]:
        return
    if seed not in plan["confirmation_seeds"]:
        raise ValueError("Seed outside the frozen protocol")
    saved = json.loads((Path(root) / "selection.json").read_text())
    expected = {key: value for key, value in saved.items() if key != "sha256"}
    if (fingerprint(expected) != saved["sha256"] or saved["plan_sha256"] != plan["sha256"]
            or saved["split_sha256"] != split_hash or trial_id not in saved["trial_ids"]):
        raise ValueError("Confirmation is not authorized by the frozen selection")
    # Recompute from ALL completed search trials, never user-selected/test-selected IDs.
    rows, search_split = ranked_search(root, plan)
    if [row["trial_id"] for row in rows[:2]] != saved["trial_ids"] or search_split != split_hash:
        raise ValueError("Search results changed after selection")
