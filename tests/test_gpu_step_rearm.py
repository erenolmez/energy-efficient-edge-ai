import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from hpc.clean_protocol import write_json, file_hash
from hpc.gpu_step_recovery import prepare as first_prepare, recovery_paths
from hpc.rearm_gpu_step import prepare, verify_rearm, rearm_paths, SOURCES, ROOT
from tests import test_gpu_step_recovery as recovery_tests


class RearmTests(unittest.TestCase):
    def fixture(self, folder, failed_text=None):
        root = folder/"study"
        root.mkdir()
        recovery_tests.RecoveryTests().fixture(root)
        with patch("hpc.gpu_step_recovery.subprocess.check_output",return_value="123\n"):
            with contextlib.redirect_stdout(io.StringIO()):
                first_prepare(root)
        write_json(recovery_paths(root)/"stop.json",
                   dict(reason="CUDA preflight failed before claiming any case",returncode=1))
        write_json(recovery_paths(root)/"submission.json",
                   dict(jobs=[dict(role="worker",job_id="110"),dict(role="worker",job_id="111")]))
        project = folder/"project"
        (project/"hpc").mkdir(parents=True)
        for name in SOURCES:
            (project/"hpc"/name).write_bytes((ROOT/"hpc"/name).read_bytes())
        for jid,node,passed in (("110","healthy1",True),("111","bad1",False)):
            events = [json.dumps(dict(event="cuda_preflight",node=node+".cluster.lan",job=jid))]
            events += [json.dumps(dict(event="cuda_preflight_passed",value=16.0))] if passed else [
                failed_text or "RuntimeError: CUDA-capable device(s) is/are busy or unavailable"]
            (project/f"grid-gpu-step-{jid}.log").write_text("\n".join(events))
        return root,project

    def test_archive_only_preflight_stop_and_keep_active_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root,project=self.fixture(Path(folder))
            original=file_hash(root/"investigate_stop.json")
            with patch("hpc.rearm_gpu_step.ROOT",project), patch(
                "hpc.rearm_gpu_step.subprocess.check_output",return_value="123\n"
            ), contextlib.redirect_stdout(io.StringIO()):
                prepare(root,["bad1"])
                receipt=verify_rearm(root)
                self.assertEqual(receipt["excluded_nodes"],["bad1"])
                self.assertEqual(len(receipt["completed_verified"]),3)
                self.assertEqual(receipt["untouched_in_flight"],[dict(case_id=3,job="123")])
                self.assertFalse((recovery_paths(root)/"stop.json").exists())
                self.assertTrue((rearm_paths(root)/"archived_preflight_stop.json").exists())
                self.assertEqual(file_hash(root/"investigate_stop.json"),original)
                self.assertTrue((root/"claims/case_0003/owner.json").exists())
                with self.assertRaises(ValueError):
                    prepare(root,["bad1"])

    def test_missing_exclusion_refuses_rearm(self):
        with tempfile.TemporaryDirectory() as folder:
            root,project=self.fixture(Path(folder))
            with patch("hpc.rearm_gpu_step.ROOT",project), patch(
                "hpc.rearm_gpu_step.subprocess.check_output",return_value="123\n"
            ), self.assertRaises(ValueError):
                prepare(root,["different"])
            self.assertTrue((recovery_paths(root)/"stop.json").exists())

    def test_active_previous_worker_refuses_rearm(self):
        with tempfile.TemporaryDirectory() as folder:
            root,project=self.fixture(Path(folder))
            with patch("hpc.rearm_gpu_step.ROOT",project), patch(
                "hpc.rearm_gpu_step.subprocess.check_output",return_value="123\n110\n"
            ), self.assertRaises(ValueError):
                prepare(root,["bad1"])

    def test_different_failure_cannot_be_rearmed(self):
        with tempfile.TemporaryDirectory() as folder:
            root,project=self.fixture(Path(folder),failed_text="unexpected permission failure")
            with patch("hpc.rearm_gpu_step.ROOT",project), patch(
                "hpc.rearm_gpu_step.subprocess.check_output",return_value="123\n"
            ), self.assertRaises(ValueError):
                prepare(root,["bad1"])
            self.assertTrue((recovery_paths(root)/"stop.json").exists())

    def test_failed_preflight_is_worker_local_without_claim(self):
        from hpc import full_factorial_worker_gpu_step_v2 as worker
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/"submission.json",dict(plan_sha256="plan",source_hashes={}))
            child=MagicMock()
            child.wait.return_value=1
            argv=["worker","--study-dir",str(root),"--data-dir","unused","--split-manifest","unused"]
            with patch.dict(os.environ,{"SLURM_JOB_ID":"999"}), patch("sys.argv",argv), patch.object(
                worker,"checked_plan",return_value=dict(sha256="plan",trials=[])
            ), patch.object(worker,"verify_rearm",return_value={}), patch.object(
                worker.subprocess,"Popen",return_value=child
            ), patch.object(worker.signal,"SIGUSR1",10,create=True), patch.object(
                worker.signal,"signal"
            ), self.assertRaises(RuntimeError):
                worker.main()
            self.assertTrue((root/"recoveries/gpu_step_v2/preflight_failures/999.json").exists())
            self.assertFalse((root/"recoveries/gpu_step_v1/stop.json").exists())
            self.assertEqual(list((root/"claims").iterdir()),[])


if __name__=="__main__":
    unittest.main()
