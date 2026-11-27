"""Dry run of hpc/gpu_feature_routers.py on CPU with synthetic evidence: ports and parity."""
import json

import numpy as np

from hpc import feature_router_ablation as fra, gpu_feature_routers as gfr, taylor_premodel as tp
from tests.test_feature_router_ablation import make_evidence
from tests.test_taylor_premodel import make_stage_a


def test_gpu_ports_dry_run(tmp_path, monkeypatch, capsys):
    ev = make_evidence(tmp_path)
    sa = make_stage_a(tmp_path, ev)
    common = ["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar")]
    fra.main(common + ["--output", str(tmp_path / "fa"), "--threads", "2", "--no-timing"])
    monkeypatch.setattr(tp, "SEEDS", (42,))
    tp.main(common + ["--stage-a", str(sa), "--output", str(tmp_path / "ta"), "--repeats", "2"])
    gfr.main(common + ["--stage-a", str(sa), "--feature-ablation", str(tmp_path / "fa"), "--taylor", str(tmp_path / "ta"),
                       "--output", str(tmp_path / "gpu"), "--device", "cpu", "--dry-run", "--threads", "2"])
    out = capsys.readouterr().out
    assert "DRY RUN OK" in out
    res = json.loads(out[out.index("{\n"):out.rindex("}") + 1])
    for k in ("compact", "texture"):
        assert res["feature_parity"][k]["max_abs"] < 1e-4
    for name, v in res["probability_parity"].items():
        # float32 feature rounding can move a value across a tree split on a rare image
        assert v["mean_abs"] < 1e-3 and v["images_over_1e3"] <= 20, (name, v)  # kNN ties too

    # end to end on CPU (no NVML): scores, cost table and routed report are written
    monkeypatch.setenv("GPU_ROUTERS_CPU_TEST", "1")
    gfr.SEEDS = (42,)
    gfr.main(common + ["--stage-a", str(sa), "--feature-ablation", str(tmp_path / "fa"), "--taylor", str(tmp_path / "ta"),
                       "--output", str(tmp_path / "gpu"), "--device", "cpu", "--threads", "2", "--min-seconds", "0.05",
                       "--warmup-seconds", "0.01", "--idle-seconds", "0", "--repeats", "1"])
    assert (tmp_path / "gpu" / "report" / "routed_report.md").exists()
    assert len(list((tmp_path / "gpu" / "scores").glob("*.npz"))) >= 8
