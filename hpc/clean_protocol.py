"""Separate, immutable CIFAR-100 roles for the new study (standard library only)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

PER_CLASS = {
    "classifier_train": 400,
    "classifier_validation": 50,
    "router_train": 30,
    "router_calibration": 20,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, writer):
    """Keep the previous checkpoint on writer/replace failure; never force-delete it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                            dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write(path, lambda temporary: temporary.write_text(data, encoding="utf-8"))


def make_split(targets, seed=42):
    targets = [int(label) for label in targets]
    if len(targets) != 50000 or set(targets) != set(range(100)):
        raise ValueError("Expected the 50,000-image official CIFAR-100 training set")
    indices = {role: [] for role in PER_CLASS}
    rng = random.Random(seed)
    for label in range(100):
        members = [index for index, target in enumerate(targets) if target == label]
        if len(members) != 500:
            raise ValueError("Expected exactly 500 training images per class")
        rng.shuffle(members)
        offset = 0
        for role, count in PER_CLASS.items():
            indices[role].extend(members[offset:offset + count])
            offset += count
    payload = {
        "schema": 1, "dataset": "CIFAR-100", "split_seed": seed,
        "index_namespace": "official_train", "targets_sha256": fingerprint(targets),
        "per_class": dict(PER_CLASS),
        "indices": {role: sorted(values) for role, values in indices.items()},
        "final_test": {"index_namespace": "official_test", "size": 10000,
                       "policy": "never instantiated or evaluated by this pilot"},
        "quantization_data_policy": "QAT and PTQ calibration use classifier_train only",
    }
    payload["sha256"] = fingerprint(payload)
    validate_split(payload, targets)
    return payload


def validate_split(manifest, targets):
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != fingerprint(payload):
        raise ValueError("Split manifest fingerprint mismatch")
    if manifest.get("per_class") != PER_CLASS:
        raise ValueError("Unexpected partition specification")
    if manifest.get("targets_sha256") != fingerprint([int(x) for x in targets]):
        raise ValueError("Dataset label/order mismatch")
    if manifest.get("index_namespace") != "official_train":
        raise ValueError("Unexpected index namespace")
    if set(manifest["indices"]) != set(PER_CLASS):
        raise ValueError("Unexpected partition names")
    seen = set()
    for role, count in PER_CLASS.items():
        values = manifest["indices"][role]
        if any(type(index) is not int or not 0 <= index < 50000 for index in values):
            raise ValueError("Invalid image index")
        if len(values) != count * 100 or len(set(values)) != len(values):
            raise ValueError("Wrong partition size or duplicate indices")
        if seen.intersection(values):
            raise ValueError("Partitions overlap")
        seen.update(values)
        counts = [0] * 100
        for index in values:
            counts[int(targets[index])] += 1
        if counts != [count] * 100:
            raise ValueError("Partition is not class-balanced")
    if seen != set(range(50000)):
        raise ValueError("Partitions do not cover the official training set")


def load_or_create_split(path, targets, seed=42):
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_split(manifest, targets)
        if manifest["split_seed"] != seed:
            raise ValueError("Existing split uses a different seed; refusing overwrite")
        return manifest
    manifest = make_split(targets, seed)
    # Exclusive creation: another preparation process must not overwrite a split.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


def validate_resume(checkpoint, recipe, split_hash, source_hashes):
    if checkpoint["recipe"] != recipe or checkpoint["split_sha256"] != split_hash:
        raise ValueError("Resume settings/split changed; use a new run directory")
    if checkpoint["source_hashes"] != source_hashes:
        raise ValueError("Resume code changed; preserve run and investigate")
    history = checkpoint["history"]
    if [row["epoch"] for row in history] != list(range(1, checkpoint["epoch"] + 1)):
        raise ValueError("Checkpoint epoch history is inconsistent")
