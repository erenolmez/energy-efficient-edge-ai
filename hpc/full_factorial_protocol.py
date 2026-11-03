"""Frozen full factorial, separate from all previously started searches."""
from __future__ import annotations
from itertools import product
import json
from pathlib import Path
from hpc.clean_protocol import fingerprint
from hpc.scratch_recipes import make_plan as old_plan

ARCHITECTURES = ("resnet18", "resnet34", "resnet50", "resnet101",
                 "mobilenetv3_small", "mobilenetv3_large",
                 "efficientnet_b0", "shufflenetv2_x1_0")


def make_plan():
    plan = old_plan()
    plan.pop("sha256")
    cases = []
    for optimizer, rates, decays in (
        ("sgd", (.03, .1, .2), (1e-4, 5e-4)),
        ("adam", (3e-4, 1e-3, 3e-3), (1e-4, 5e-4)),
        ("adamw", (3e-4, 1e-3, 3e-3), (.01, .05)),
    ):
        for rate, decay, schedule, cutmix, seed, architecture in product(
            rates, decays, ("cosine", "step80_120"), (0.0, 1.0),
            (42, 45, 48), ARCHITECTURES,
        ):
            cases.append(dict(id=len(cases), optimizer=optimizer, lr=rate,
                              weight_decay=decay, schedule=schedule,
                              cutmix_alpha=cutmix, seed=seed, architecture=architecture))
    plan.update(study="scratch32_full_factorial_v1", architecture="mixed",
                architectures=list(ARCHITECTURES), trials=cases, seeds=[42, 45, 48],
                maximum_training_epochs=len(cases)*200,
                selection="All configurations use all three seeds; rank mean best classifier-validation accuracy",
                final_selection="Mean seed42/45/48 validation accuracy; loss then config key break ties",
                qualification="Complete grid only over the declared finite candidates, not every possible hyperparameter. "
                "Torchvision implementations with documented CIFAR stems, not identical to historical timm/128px models.",
                stem="ResNet: 3x3 stride1/no maxpool. Others: first convolution stride1; "
                     "ShuffleNet also removes initial maxpool; later blocks unchanged.")
    for key in ("confirmation_seeds", "confirmation_finalists", "search_seed"):
        plan.pop(key, None)
    plan["sha256"] = fingerprint(plan)
    return plan


def checked_plan(path):
    actual = json.loads(Path(path).read_text())
    if actual != make_plan():
        raise ValueError("Full-factorial frozen plan changed")
    return actual


def freeze_plan(path):
    path = Path(path)
    if path.exists():
        return checked_plan(path)
    plan = make_plan()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def trial_directory(root, trial_id, seed):
    if trial_id not in range(1728) or seed not in (42, 45, 48):
        raise ValueError("Unknown case")
    return Path(root) / f"case_{trial_id:04d}" / f"seed_{seed}"


def build_model(architecture):
    import torch
    from torchvision import models
    constructors = dict(resnet18=models.resnet18, resnet34=models.resnet34,
                        resnet50=models.resnet50, resnet101=models.resnet101,
                        mobilenetv3_small=models.mobilenet_v3_small,
                        mobilenetv3_large=models.mobilenet_v3_large,
                        efficientnet_b0=models.efficientnet_b0,
                        shufflenetv2_x1_0=models.shufflenet_v2_x1_0)
    model = constructors[architecture](weights=None, num_classes=100)
    if architecture.startswith("resnet"):
        model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        model.maxpool = torch.nn.Identity()
    elif architecture == "shufflenetv2_x1_0":
        model.conv1[0].stride = (1, 1)
        model.maxpool = torch.nn.Identity()
    else:
        model.features[0][0].stride = (1, 1)
    return model
