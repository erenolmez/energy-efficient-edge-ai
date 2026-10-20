import sys
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pc_tests/model_pipeline'))
from run_most_pruned import highest_pruned
from run_most_pruned_batches import BATCH_SIZES, read_thermal_state
from batch_router_policy import make_batches


class MostPrunedBatchTests(unittest.TestCase):
    def test_transient_empty_or_partial_status_is_retried(self):
        path=Mock()
        path.read_text.side_effect=['', '{', '{"status":"monitoring"}']
        with patch('run_most_pruned_batches.time.sleep'):
            self.assertEqual(read_thermal_state(path)['status'],'monitoring')
        self.assertEqual(path.read_text.call_count,3)

    def test_persistent_unreadable_status_fails_closed(self):
        path=Mock()
        path.read_text.return_value=''
        with patch('run_most_pruned_batches.time.sleep'):
            with self.assertRaises(RuntimeError):
                read_thermal_state(path,attempts=3)
        self.assertEqual(path.read_text.call_count,3)

    def test_latched_status_is_not_replaced_or_retried(self):
        path=Mock()
        path.read_text.return_value='{"status":"thermal_stop_shutdown_requested"}'
        self.assertEqual(read_thermal_state(path)['status'],'thermal_stop_shutdown_requested')
        self.assertEqual(path.read_text.call_count,1)

    def test_real_batch_shapes_and_coverage(self):
        for size,count,tail in ((8,1250,8),(32,313,16),(64,157,16)):
            batches=make_batches(np.arange(10000),size,42+size)
            self.assertEqual(len(batches),count)
            self.assertEqual(len(batches[-1]),tail)
            self.assertEqual(len(np.unique(np.concatenate(batches))),10000)
        self.assertEqual(BATCH_SIZES,(8,32,64))

    def test_highest_pruning_ignores_test_accuracy(self):
        records=[dict(architecture=f'a{i}',prune=p,precision=q,test_accuracy=100-p)
                 for i in range(7) for p in (0,30,90) for q in ('fp32','fp16','int8')]
        records.append(dict(architecture='unpruned_only',prune=0,precision='fp32',test_accuracy=100))
        selected=highest_pruned(records)
        self.assertEqual(len(selected),21)
        self.assertTrue(all(r['prune']==90 for r in selected))
        for r in records:
            r['test_accuracy']=-999
        self.assertEqual([r['architecture'] for r in highest_pruned(records)], [r['architecture'] for r in selected])


if __name__=='__main__':
    unittest.main()
