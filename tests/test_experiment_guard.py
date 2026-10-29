import json
import os
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest.mock import patch
from pc_tests.model_pipeline.runtime import experiment_guard as guard

class ExperimentGuardTests(unittest.TestCase):
    def test_identity_pause_stale_and_latch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'reports').mkdir()
            (root/'artifacts/pc_routers').mkdir(parents=True)
            pause=root/'reports/experiment_pause.json'
            path=root/'artifacts/pc_routers/thermal_guard.json'
            pause.write_text(json.dumps({'status':'resumed_by_user'}))
            state=dict(status='monitoring',boundary_start_count=None,utc=datetime.now(timezone.utc).isoformat(),
                       expected_script='worker.py',tracked_process_ids=[os.getpid()])
            with patch.object(guard,'PIPELINE',root):
                path.write_text(json.dumps(state));self.assertEqual(guard.check('worker.py')['status'],'monitoring')
                for key,value in [('expected_script','other.py'),('tracked_process_ids',[]),('status','waiting_for_next_checkpoint'),
                                  ('boundary_start_count',0),('utc',(datetime.now(timezone.utc)-timedelta(seconds=131)).isoformat())]:
                    changed=dict(state);changed[key]=value;path.write_text(json.dumps(changed))
                    with self.assertRaises(RuntimeError): guard.check('worker.py')
                path.write_text(json.dumps(state));pause.write_text(json.dumps({'status':'paused_by_user'}))
                with self.assertRaises(RuntimeError): guard.check('worker.py')

if __name__=='__main__': unittest.main()
