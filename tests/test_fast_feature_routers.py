"""hpc/fast_feature_routers.py on CPU with synthetic evidence: dry run, then end to end (no NVML)."""
import json

import numpy as np

from hpc import fast_feature_routers as ffr, feature_router_ablation as fra, taylor_premodel as tp
from tests.test_feature_router_ablation import make_evidence
from tests.test_taylor_premodel import make_stage_a


def test_folds_are_stratified():
    labels = np.repeat(np.arange(10), 50)
    pool = np.arange(500)
    fold = ffr.make_folds(pool, labels)
    for f in range(5):
        assert (np.bincount(labels[fold == f], minlength=10) == 10).all()


def test_fast_routers_cpu(tmp_path, monkeypatch, capsys):
    ev = make_evidence(tmp_path)
    sa = make_stage_a(tmp_path, ev)
    common = ["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar")]
    fra.main(common + ["--output", str(tmp_path / "fa"), "--threads", "2", "--no-timing"])
    monkeypatch.setattr(tp, "SEEDS", (42,))
    tp.main(common + ["--stage-a", str(sa), "--output", str(tmp_path / "ta"), "--repeats", "2"])
    args = common + ["--stage-a", str(sa), "--feature-ablation", str(tmp_path / "fa"), "--taylor", str(tmp_path / "ta"),
                     "--output", str(tmp_path / "fast"), "--device", "cpu", "--threads", "2"]
    capsys.readouterr()
    ffr.main(args + ["--dry-run"])
    out = capsys.readouterr().out
    assert "DRY RUN OK" in out
    res = json.loads(out[out.index("{\n"):out.rindex("}") + 1])
    assert res["components"]["images_differing"] == 0
    for k in ("compact", "texture"):
        assert res["feature_parity"][k]["max_abs"] < 1e-4
    assert res["feature_parity"]["taylor"]["images_any_mismatch"] == 0, res["feature_parity"]["taylor"]
    assert not (tmp_path / "fast").exists()

    monkeypatch.setenv("GPU_ROUTERS_CPU_TEST", "1")
    monkeypatch.setattr(ffr, "SEEDS", (42,))
    ffr.main(args + ["--min-seconds", "0.02", "--warmup-seconds", "0.01", "--idle-seconds", "0", "--repeats", "1"])
    files = sorted((tmp_path / "fast" / "scores").glob("*.npz"))
    assert len(files) == 5
    split = json.loads((tmp_path / "split_seed42.json").read_text())["indices"]
    pool = np.concatenate([split["router_train"], split["router_calibration"]])
    for f in files:
        z = np.load(f)
        assert (z["index"] == pool).all() and z["correct_prob"].shape == (5000, 8)
        assert np.isfinite(z["correct_prob"]).all() and set(np.unique(z["fold"])) == set(range(5))
        assert (z["router_latency_ms"] > 0).all() and len(z["router_cost_batches"]) == 5
    assert not np.intersect1d(pool, split["classifier_validation"]).size
