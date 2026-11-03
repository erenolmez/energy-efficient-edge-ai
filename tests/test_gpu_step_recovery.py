import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import make_plan, trial_directory
from hpc.gpu_step_recovery import ROOT, prepare, recovery_paths, verify_authority


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        plan = make_plan()
        write_json(root/"plan.json", plan)
        original_names = ("full_factorial_protocol.py", "train_full_factorial.py",
                         "full_factorial_worker.py", "full_factorial.sbatch", "clean_protocol.py",
                         "scratch_recipes.py", "train_classifier_pilot.py", "wandb_tracking.py")
        sources = {name: file_hash(ROOT/"hpc"/name) for name in original_names}
        write_json(root/"submission.json", dict(plan_sha256=plan["sha256"], source_hashes=sources))
        write_json(root/"data_ready.json", dict(plan_sha256=plan["sha256"]))
        write_json(root/"investigate_stop.json", dict(case_id=2, returncode=1))
        trainer_names = ("train_full_factorial.py", "full_factorial_protocol.py", "clean_protocol.py",
                         "scratch_recipes.py", "train_classifier_pilot.py", "wandb_tracking.py")
        for case_id in (0, 1, 2, 3, 4):
            case = plan["trials"][case_id]
            claim = root/"claims"/f"case_{case_id:04d}"
            claim.mkdir(parents=True)
            write_json(claim/"owner.json", dict(case_id=case_id, plan_sha256=plan["sha256"],
                                               job="123" if case_id == 3 else "999"))
            directory = trial_directory(root, case_id, case["seed"])
            directory.mkdir(parents=True)
            if case_id == 3:
                (directory/"train.log").write_text("ongoing unchanged")
                continue
            if case_id == 2:
                write_json(claim/"result.json", dict(status="failed", epochs_complete=None, returncode=1))
                (directory/"train.log").write_text(
                    "model = model.cuda()\nRuntimeError: CUDA-capable device(s) is/are busy or unavailable")
                continue
            (directory/"latest.pt").write_bytes(b"synthetic-checkpoint")
            history = [dict(epoch=e, train=dict(images=40000, accuracy=90, loss=.2, seconds=1),
                            validation=dict(images=5000, accuracy=70, loss=1, seconds=.1))
                       for e in range(1, 201)]
            write_json(directory/"summary.json", dict(status="complete", epochs_complete=200,
                       trial=case, seed=case["seed"], plan_sha256=plan["sha256"],
                       test_evaluated=False, checkpoint_sha256=file_hash(directory/"latest.pt"),
                       source_hashes={"hpc/"+name:sources[name] for name in trainer_names},
                       history=history))
            write_json(claim/"result.json", dict(status="complete", epochs_complete=200))
        return plan

    @patch("hpc.gpu_step_recovery.subprocess.check_output", return_value="123\n")
    def test_verified_archive_preserves_completed_active_and_marker(self, _):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            stop_hash = file_hash(root/"investigate_stop.json")
            complete_hash = file_hash(trial_directory(root,0,42)/"summary.json")
            prepare(root)
            receipt = verify_authority(root)
            self.assertEqual(len(receipt["completed_verified"]), 3)
            self.assertEqual(receipt["untouched_in_flight"], [dict(case_id=3, job="123")])
            self.assertEqual(file_hash(root/"investigate_stop.json"), stop_hash)
            self.assertEqual(file_hash(trial_directory(root,0,42)/"summary.json"), complete_hash)
            self.assertEqual((trial_directory(root,3,42)/"train.log").read_text(), "ongoing unchanged")
            self.assertTrue((recovery_paths(root)/"failed_case_0002/train.log").exists())
            self.assertFalse((root/"claims/case_0002").exists())
            with self.assertRaises((ValueError, FileNotFoundError)):
                prepare(root)
            write_json(root/"investigate_stop.json", dict(case_id=5, returncode=1))
            with self.assertRaises(ValueError):
                verify_authority(root)

    @patch("hpc.gpu_step_recovery.subprocess.check_output", return_value="123\n")
    def test_checkpoint_in_failed_case_blocks_archive(self, _):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (trial_directory(root,2,42)/"latest.pt").write_bytes(b"must-not-move")
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertTrue((root/"claims/case_0002").exists())

    @patch("hpc.gpu_step_recovery.subprocess.check_output", return_value="123\n")
    def test_bad_completed_checkpoint_blocks(self, _):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (trial_directory(root,0,42)/"latest.pt").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertFalse((recovery_paths(root)/"receipt.json").exists())

    @patch("hpc.gpu_step_recovery.subprocess.check_output", return_value="123\n999\n")
    def test_failed_owner_still_active_blocks(self, _):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertTrue((trial_directory(root,2,42)/"train.log").exists())


if __name__ == "__main__":
    unittest.main()
