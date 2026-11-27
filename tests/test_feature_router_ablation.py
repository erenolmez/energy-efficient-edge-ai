"""Smoke test for hpc/feature_router_ablation.py on synthetic evidence (no HPC data)."""
import json

import numpy as np

from hpc.clean_protocol import make_split
from hpc import feature_router_ablation as fra

NAMES = ["resnet18", "resnet34", "resnet50", "resnet101", "mobilenetv3_small",
         "mobilenetv3_large", "efficientnet_b0", "shufflenetv2_x1_0"]


def make_evidence(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(100), 500)
    rng.shuffle(labels)
    split = make_split(labels.tolist())
    (tmp_path / "split_seed42.json").write_text(json.dumps(split))
    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / "split.json").write_text(json.dumps({"sha256": split["sha256"]}))
    (ev / "bank.json").write_text(json.dumps([{"architecture": n} for n in NAMES]))
    np.save(ev / "labels.npy", labels)
    images = np.zeros((50000, 32, 32, 3), np.uint8)
    used = np.concatenate([split["indices"][k] for k in
                           ("classifier_validation", "router_train", "router_calibration")])
    brightness = rng.integers(0, 256, size=50000)
    images[used] = np.clip(brightness[used, None, None, None]
                           + rng.integers(-20, 20, size=(len(used), 32, 32, 3)), 0, 255)
    # native 32px pixels live in the CIFAR-100 python pickle (evidence images.npy is the 128px resize)
    import pickle
    (tmp_path / "cifar" / "cifar-100-python").mkdir(parents=True)
    with open(tmp_path / "cifar" / "cifar-100-python" / "train", "wb") as f:
        pickle.dump({b"data": images.transpose(0, 3, 1, 2).reshape(50000, -1),
                     b"fine_labels": labels.tolist()}, f)
    # Correctness depends on brightness, so features should beat the constant baseline.
    for j, n in enumerate(NAMES):
        p_ok = 0.3 + 0.6 * (brightness / 255) * (0.5 + j / 16)
        ok = rng.random(50000) < p_ok
        np.save(ev / f"{n}.npy", np.where(ok, labels, (labels + 1) % 100))
    return ev


def test_quick_run(tmp_path):
    ev = make_evidence(tmp_path)
    out = tmp_path / "out"
    fra.main(["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar"), "--dry-run",
              "--output", str(tmp_path / "unused")])
    assert not (tmp_path / "unused").exists()
    fra.main(["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar"), "--output", str(out), "--quick", "--threads", "2"])
    result = json.loads((out / "results.json").read_text())
    s = result["summary"]
    assert result["score_size"] == 2000 and result["fit_sets"] == {"val3k": 8000, "3k": 3000}
    assert s["compact_logistic@val3k"]["mean_brier"] < s["constant@val3k"]["mean_brier"]
    assert s["compact_hgb_cond@val3k"]["seeds"] == 3
    assert (out / "summary.md").read_text().startswith("# Hand-made features")
