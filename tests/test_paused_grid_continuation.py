"""CPU-only tests: no scheduler, dataset, W&B API, or training access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpc import continue_paused_grid_20261104 as continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "case_0154/seed_45"
        self.claim = self.root / "claims/case_0154"
        self.directory.mkdir(parents=True)
        self.claim.mkdir(parents=True)
        self.case = dict(id=154, seed=45, architecture="resnet50")
        self.plan = dict(trials=[None] * 154 + [self.case])
        (self.directory / "latest.pt").write_bytes(b"owned-test-checkpoint")
        self.saved = dict(status="paused", epochs_complete=178, trial=self.case, seed=45,
                          plan_sha256=continuation.PLAN, test_evaluated=False,
                          source_hashes={}, checkpoint_sha256=continuation.file_hash(self.directory / "latest.pt"))
        self.result = dict(status="paused", epochs_complete=178, returncode=0)
        self.owner = dict(case_id=154, plan_sha256=continuation.PLAN)
        self.save()

    def put(self, path, value):
        path.write_text(json.dumps(value))

    def save(self):
        self.put(self.directory / "summary.json", self.saved)
        self.put(self.directory / "run.json", dict(source_hashes={}))
        self.put(self.claim / "result.json", self.result)
        self.put(self.claim / "owner.json", self.owner)
        self.put(self.directory / "wandb_identity.json", dict(id="existing-run", entity="erenolmez-politecnico-di-torino", project="energy-efficient-edge-ai"))
        self.put(self.directory / "wandb_status.json", dict(run_id="existing-run"))

    def check(self):
        return continuation.check_case(self.root, self.plan, 154)

    def test_valid_paused_case_is_read_only(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.check()
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_stale_or_live_lock_never_removed(self):
        lock = self.directory / "active.lock"
        lock.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "lock"):
            self.check()
        self.assertEqual(lock.read_text(), "preserve")

    def test_changed_epoch_rejected(self):
        self.saved["epochs_complete"] = 179
        self.save()
        with self.assertRaisesRegex(ValueError, "epoch"):
            self.check()

    def test_failed_case_not_retried(self):
        self.result["status"] = "failed"
        self.save()
        with self.assertRaisesRegex(ValueError, "paused"):
            self.check()

    def test_changed_checkpoint_rejected(self):
        (self.directory / "latest.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.check()

    def test_wrong_plan_rejected(self):
        self.owner["plan_sha256"] = "wrong"
        self.save()
        with self.assertRaisesRegex(ValueError, "Plan"):
            self.check()

    def test_changed_tracking_identity_rejected(self):
        self.put(self.directory / "wandb_status.json", dict(run_id="different"))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.check()

    def test_shared_v1_stop_is_honored(self):
        stop = self.root / "recoveries/gpu_step_v1/stop.json"
        stop.parent.mkdir(parents=True)
        stop.write_text("{}")
        with patch.object(continuation, "verify_rearm"), self.assertRaisesRegex(ValueError, "safety stop"):
            continuation.gate(self.root)

    def test_existing_failure_evidence_not_overwritten(self):
        stop = self.root / "recoveries/gpu_step_v1/stop.json"
        stop.parent.mkdir(parents=True)
        stop.write_text('{"reason":"original failure"}')
        continuation.latch(self.root, 154, "new failure")
        self.assertEqual(stop.read_text(), '{"reason":"original failure"}')

    def test_submission_holds_and_records_all_jobs_before_release(self):
        calls = []

        def output(command, **kwargs):
            calls.append(command)
            if command[0] == "squeue":
                return ""
            if command[0] == "scontrol":
                return "UserName=tester(123) Partition=gpu_a40"
            self.assertIn("--hold", command)
            return str(9000 + sum(c[0] == "sbatch" for c in calls))

        def release(command):
            receipt = continuation.read(self.root / "recoveries" / continuation.BATCH / "submission.json")
            self.assertEqual(receipt["status"], "releasing_verified_jobs")
            self.assertEqual(len(receipt["jobs"]), 4)
            calls.append(command)

        (self.root / "recoveries").mkdir()
        with patch.object(continuation, "tracking_gate"), patch.object(continuation, "gate", return_value=self.plan), \
             patch.object(continuation, "check_case"), patch.object(continuation.getpass, "getuser", return_value="tester"), \
             patch.object(continuation.subprocess, "check_output", side_effect=output), \
             patch.object(continuation.subprocess, "check_call", side_effect=release), \
             patch("builtins.print"):
            continuation.submit(self.root, "gpu_a40", True)
        self.assertEqual(sum(c[0] == "sbatch" for c in calls), 4)
        self.assertEqual(sum(c[:2] == ["scontrol", "release"] for c in calls), 4)
        receipt = continuation.read(self.root / "recoveries" / continuation.BATCH / "submission.json")
        self.assertEqual(receipt["status"], "submitted")
        self.assertTrue(all(j["released"] for j in receipt["jobs"]))


if __name__ == "__main__":
    unittest.main()
