"""Model registry used by PC training and Jetson deployment."""
from __future__ import annotations

from dataclasses import dataclass

import timm
import torch.nn as nn
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0


@dataclass(frozen=True)
class Architecture:
    name: str
    timm_name: str
    family: str
    source: str = "timm"


ARCHITECTURES = {
    "resnet18": Architecture("resnet18", "resnet18", "resnet"),
    "resnet34": Architecture("resnet34", "resnet34", "resnet"),
    "resnet50": Architecture("resnet50", "resnet50", "resnet"),
    "resnet101": Architecture("resnet101", "resnet101", "resnet"),
    "mobilenetv3_small": Architecture(
        "mobilenetv3_small", "mobilenetv3_small_100", "mobilenetv3"
    ),
    "mobilenetv3_large": Architecture(
        "mobilenetv3_large", "mobilenetv3_large_100", "mobilenetv3"
    ),
    "efficientnet_b0": Architecture("efficientnet_b0", "efficientnet_b0", "efficientnet"),
    "shufflenetv2_x1_0": Architecture(
        "shufflenetv2_x1_0", "shufflenet_v2_x1_0", "shufflenetv2", "torchvision"
    ),
}

ALIASES = {
    "mobilenetv3-small": "mobilenetv3_small",
    "mobilenetv3-large": "mobilenetv3_large",
    "efficientnet-b0": "efficientnet_b0",
    "shufflenetv2": "shufflenetv2_x1_0",
    "shufflenet_v2_x1_0": "shufflenetv2_x1_0",
}


def normalize_architecture(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in ARCHITECTURES:
        supported = ", ".join(ARCHITECTURES)
        raise ValueError(f"Unsupported architecture {name!r}. Choose from: {supported}")
    return normalized


def create_classifier(
    architecture: str,
    *,
    num_classes: int = 100,
    pretrained: bool = False,
):
    """Create one of the experiment classifiers with a common output size."""
    architecture = normalize_architecture(architecture)
    definition = ARCHITECTURES[architecture]
    if definition.source == "torchvision":
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return timm.create_model(
        definition.timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
