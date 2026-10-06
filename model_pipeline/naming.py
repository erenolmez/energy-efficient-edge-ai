"""Unambiguous names for architecture, pruning and precision variants."""
from __future__ import annotations

import re

from .models import normalize_architecture

MODEL_NAME = re.compile(
    r"^(?P<architecture>[a-z0-9_]+)__p(?P<prune>\d{1,2})(?:__(?P<precision>fp32|fp16|int8))?$"
)


def model_name(architecture: str, prune: int = 0, precision: str | None = None) -> str:
    architecture = normalize_architecture(architecture)
    if prune < 0 or prune >= 100:
        raise ValueError("prune must be in [0, 100)")
    result = f"{architecture}__p{prune}"
    if precision is not None:
        precision = precision.lower()
        if precision not in {"fp32", "fp16", "int8"}:
            raise ValueError(f"Unsupported precision: {precision}")
        result += f"__{precision}"
    return result


def parse_model_name(value: str):
    match = MODEL_NAME.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid model name: {value}")
    fields = match.groupdict()
    fields["architecture"] = normalize_architecture(fields["architecture"])
    fields["prune"] = int(fields["prune"])
    return fields
