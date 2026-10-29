import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

class AtomicCheckpointTests(unittest.TestCase):
    def test_success(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';atomic_json(path,{'saved':41})
            self.assertEqual(json.loads(path.read_text()),{'saved':41})
    def test_transient_reader_lock(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';atomic_json(path,{'saved':41});original=Path.replace;calls=[]
            def replace(tmp,target):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(path.read_text()),{'saved':41})
                    raise PermissionError('simulated Windows reader lock')
                return original(tmp,target)
            with patch.object(Path,'replace',replace):atomic_json(path,{'saved':42},attempts=3,delay_seconds=0)
            self.assertEqual(len(calls),3);self.assertEqual(json.loads(path.read_text()),{'saved':42})
    def test_persistent_lock_preserves_destination_and_new_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';atomic_json(path,{'saved':41})
            with patch.object(Path,'replace',side_effect=PermissionError('persistent')) as replace:
                with self.assertRaises(PermissionError):atomic_json(path,{'saved':42},attempts=3,delay_seconds=0)
            self.assertEqual(replace.call_count,3);self.assertEqual(json.loads(path.read_text()),{'saved':41})
            self.assertEqual(len(list(Path(d).glob('state.json.tmp.*'))),1)
    def test_other_errors_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json'
            with patch.object(Path,'replace',side_effect=OSError('disk fault')) as replace:
                with self.assertRaises(OSError):atomic_json(path,{})
                self.assertEqual(replace.call_count,1)
    def test_nonfinite_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';atomic_json(path,{'saved':41})
            with self.assertRaises(ValueError):atomic_json(path,{'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'saved':41})

if __name__=='__main__':unittest.main()
