"""routed_report on synthetic evidence: feature-ablation probabilities and a40_policy-format scores."""
import numpy as np

from hpc import feature_router_ablation as fra, routed_report
from tests.test_feature_router_ablation import NAMES, make_evidence
from tests.test_taylor_premodel import make_stage_a


def test_routed_report(tmp_path):
    ev = make_evidence(tmp_path)
    sa = make_stage_a(tmp_path, ev)
    fa = tmp_path / "fa"
    fra.main(["--base", str(tmp_path), "--evidence", str(ev), "--data", str(tmp_path / "cifar"),
              "--output", str(fa), "--quick", "--threads", "2"])
    z = np.load(sa / "predictions.npz")
    sc = tmp_path / "direct-x-s42.npz"
    np.savez(sc, index=z["index"], models=np.array(NAMES[::-1]), correct_prob=np.full((2000, 8), 0.7))
    out = tmp_path / "rep"
    routed_report.main(["--stage-a", str(sa), "--scores", str(sc), "--feature-ablation", str(fa),
                        "--feature-methods", "compact_hgb_cond", "compact_logistic", "--output", str(out),
                        "--repeats", "3"])
    md = (out / "routed_report.md").read_text()
    assert "compact_logistic-val3k" in md and "compact_hgb_cond-3k-s42" in md and "direct-x" in md
    assert "latency saving" in md
