import json
from pathlib import Path
import tempfile
import unittest

from hpc.scratch_recipes import make_plan
from hpc.wandb_tracking import public_config, public_metrics, tracking_session


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logs = []
        self.finish_code = None

    def log(self, metrics, step):
        self.logs.append((metrics, step))

    def finish(self, exit_code):
        self.finish_code = exit_code


class FakeSDK:
    __version__ = "0.23.0"

    def __init__(self):
        self.run = FakeRun()
        self.options = None

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.options = kwargs
        return self.run


class TrackingTests(unittest.TestCase):
    def config(self):
        plan = make_plan()
        return public_config(plan, plan["trials"][0], 42, "search")

    def row(self):
        return {"epoch": 1, "lr": 0.1, "train": {"accuracy": 30, "loss": 2, "seconds": 10},
                "validation": {"accuracy": 40, "loss": 1, "seconds": 2},
                "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                "hostname": "must-not-be-logged", "api_key": "must-not-be-logged"}

    def test_disabled_never_initializes_sdk(self):
        with tracking_session(None, self.config(), mode="disabled", sdk=object()) as tracker:
            tracker.log(self.row())

    def test_allowlists_do_not_forward_private_fields(self):
        plan = make_plan()
        plan["hostname"] = "private-host"
        plan["api_key"] = "private-secret"
        config = public_config(plan, plan["trials"][0], 42, "search")
        self.assertNotIn("hostname", config)
        self.assertNotIn("api_key", config)
        metrics = public_metrics(self.row())
        self.assertNotIn("hostname", metrics)
        self.assertNotIn("api_key", metrics)

    def test_online_privacy_settings_logging_and_stable_id(self):
        with tempfile.TemporaryDirectory() as folder:
            sdk = FakeSDK()
            with tracking_session(folder, self.config(), "online", "private-project", "account", sdk) as tracker:
                tracker.log(self.row())
            settings = sdk.options["settings"]
            for field in ("disable_git", "disable_code", "x_disable_meta", "x_disable_stats", "x_disable_machine_info"):
                self.assertTrue(settings[field])
            self.assertEqual(settings["console"], "off")
            self.assertFalse(settings["save_code"])
            self.assertEqual(sdk.options["resume"], "allow")
            self.assertEqual(sdk.run.logs[0][1], 1)
            self.assertEqual(sdk.run.finish_code, 0)
            first_id = sdk.options["id"]
            again = FakeSDK()
            with tracking_session(folder, self.config(), "online", "private-project", "account", again):
                pass
            self.assertEqual(first_id, again.options["id"])

    def test_offline_segments_have_separate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = FakeSDK(), FakeSDK()
            for sdk in (a, b):
                with tracking_session(folder, self.config(), "offline", "project", sdk=sdk):
                    pass
            self.assertNotEqual(a.options["id"], b.options["id"])
            self.assertEqual(a.options["config"]["logical_run_id"], b.options["config"]["logical_run_id"])
            self.assertIsNone(a.options["resume"])

    def test_missing_destination_and_changed_destination_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for mode, project, entity in (("online", "project", None), ("offline", None, None)):
                with self.assertRaises(ValueError):
                    with tracking_session(folder, self.config(), mode, project, entity, FakeSDK()):
                        pass
            with tracking_session(folder, self.config(), "online", "first", "account", FakeSDK()):
                pass
            with self.assertRaises(ValueError):
                with tracking_session(folder, self.config(), "online", "other", "account", FakeSDK()):
                    pass

    def test_log_failure_preserves_training_and_records_generic_error(self):
        with tempfile.TemporaryDirectory() as folder:
            sdk = FakeSDK()
            def fail(*args, **kwargs):
                raise RuntimeError("private-host token=do-not-print")
            sdk.run.log = fail
            with tracking_session(folder, self.config(), "offline", "project", sdk=sdk) as tracker:
                tracker.log(self.row())
            status = (Path(folder) / "wandb_status.json").read_text()
            self.assertNotIn("do-not-print", status)
            self.assertEqual(json.loads(status)["status"], "logging_failed")

    def test_exception_finishes_as_failed_without_suppressing_it(self):
        with tempfile.TemporaryDirectory() as folder:
            sdk = FakeSDK()
            with self.assertRaisesRegex(ValueError, "training failure"):
                with tracking_session(folder, self.config(), "offline", "project", sdk=sdk):
                    raise ValueError("training failure")
            self.assertEqual(sdk.run.finish_code, 1)


if __name__ == "__main__":
    unittest.main()
