"""Smoke test for hpc/taylor_premodel.py on synthetic evidence and a synthetic stage A."""
import json

import numpy as np

from hpc import taylor_premodel as tp
from tests.test_feature_router_ablation import NAMES, make_evidence


def make_stage_a(tmp_path, ev):
    split = json.loads((tmp_path / "split_seed42.json").read_text())
    cal = np.array(split["indices"]["router_calibration"])
    labels = np.load(ev / "labels.npy")
    preds = np.stack([np.load(ev / f"{n}.npy")[cal] for n in NAMES])
    sa = tmp_path / "stage_a"
    sa.mkdir()
    np.savez(sa / "predictions.npz", models=np.array(NAMES), roles=np.array(["router_calibration"]),
             role=np.zeros(len(cal), int), pred_trt_fp16_gpu=preds, label=labels[cal], index=cal)
    rows = [dict(model=n, variant="trt_fp16", batch=b, latency_ms=b * (1 + j), energy_mj=b * (2 + j))
            for j, n in enumerate(NAMES) for b in (8, 16, 32, 64, 128)]
    (sa / "costs_pipeline.json").write_text(json.dumps(dict(meta=dict(test_evaluated=False, device=dict(job="x")), rows=rows)))
    return sa


def test_premodel_quick(tmp_path, monkeypatch):
    ev = make_evidence(tmp_path)
    sa = make_stage_a(tmp_path, ev)
    monkeypatch.setattr(tp, "SEEDS", (42,))
    out = tmp_path / "out"
    common = ["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar"), "--stage-a", str(sa)]
    tp.main(common + ["--output", str(tmp_path / "unused"), "--dry-run"])
    assert not (tmp_path / "unused").exists()
    tp.main(["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar"), "--stage-a", str(sa), "--output", str(out),
             "--repeats", "3"])
    r = json.loads((out / "results.json").read_text())
    assert len(r["feature_sets"]["sel-s42"]["selected"]) == 7
    assert r["chains"]["latency"]["models"]
    z = np.load(out / "scores" / "taylor-knn8-paper7.npz")
    assert z["correct_prob"].shape == (2000, 8)
    from hpc import a40_policy
    a40_policy.main(["--stage-a", str(sa), "--scores", str(out / "scores" / "taylor-knn8-paper7.npz"),
                     "--output", str(out / "policy"), "--repeats", "3"])
    assert (out / "policy" / "policy.md").exists()
