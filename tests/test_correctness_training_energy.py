import json
import os
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pc_tests.model_pipeline.experiments.measure_correctness_training_energy import read_guard, require_energy


class TrainingEnergyTests(unittest.TestCase):
    def test_missing_zero_negative_energy_rejected(self):
        for value in (None, 0, -1, float('nan'), float('inf')):
            with self.assertRaises(RuntimeError):
                require_energy({'gpu_energy_j': value})

    def test_positive_energy(self):
        self.assertEqual(require_energy({'gpu_energy_j': 12})['gpu_energy_j'], 12)

    def test_guard_identity_freshness_and_latch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            now = datetime.now(timezone.utc)
            state = dict(status='monitoring', utc=now.isoformat(), tracked_process_ids=[os.getpid()],
                         expected_script='measure_correctness_training_energy.py', training_marker_directory=directory)
            path.write_text(json.dumps(state))
            self.assertEqual(read_guard(path, Path(directory), now)['status'], 'monitoring')
            for key,value in [('status','waiting_for_next_checkpoint'), ('tracked_process_ids',[]),
                              ('expected_script','other.py'), ('utc',(now-timedelta(seconds=131)).isoformat()),
                              ('training_marker_directory',directory+'/wrong')]:
                bad = dict(state, **{key:value})
                path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):
                    read_guard(path, Path(directory), now)
