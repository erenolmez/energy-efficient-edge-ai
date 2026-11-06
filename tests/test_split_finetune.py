import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pc_tests.model_pipeline.scripts import split_finetune as handoff


class SplitFinetuneTest(unittest.TestCase):
    def test_tail_keeps_running_case_on_pc(self):
        cases = [dict(id=f'resnet18__p{i}', architecture='resnet18') for i in range(10, 100, 10)]
        done = {'resnet18__p10'}
        shards = handoff.allocate_tail(cases, done, 'resnet18__p20', 3, 6)
        self.assertEqual(shards['pc'], ['resnet18__p20', 'resnet18__p30'])
        self.assertEqual({i for k, ids in shards.items() if k != 'pc' for i in ids},
                         {c['id'] for c in cases[-6:]})
        with self.assertRaisesRegex(RuntimeError, 'already reached'):
            handoff.allocate_tail(cases, done, 'resnet18__p40', 3, 6)

    def test_assignment(self):
        cases = [dict(id=f'{arch}__p{i}', architecture=arch)
                 for arch in handoff.ESTIMATE for i in range(10, 100, 10)]
        done = {'resnet18__p10', 'resnet18__p20'}
        interrupted = 'resnet34__p10'
        shards = handoff.allocate(cases, done, interrupted, 3)
        assigned = [cid for ids in shards.values() for cid in ids]
        self.assertEqual(len(assigned), 61)
        self.assertEqual(set(assigned), {case['id'] for case in cases} - done)
        self.assertIn(interrupted, shards['pc'])

    def test_collect_refuses_corruption_and_then_indexes(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            pc = base / 'pc-results'
            remote = base / 'hpc-results'
            pc.mkdir()
            remote.mkdir()
            cases = [dict(id=f'resnet18__p{i}', architecture='resnet18', level=i, batch=64)
                     for i in (10, 20)]
            plan = base / 'handoff.json'
            recipe = dict(split_sha256='split', stop=dict(min_epochs=10, max_epochs=25,
                          patience=5, min_delta_pp=.2))
            handoff.write(plan, dict(cases=cases, recipe=recipe, prior_ids=[],
                                     shards={'pc': [cases[0]['id']], 'hpc0': [cases[1]['id']]}))
            handoff.write(base / 'prior_pc.json', {})
            for owner, case, root in zip(('pc', 'hpc0'), cases, (pc, remote)):
                model = root / 'input.pth'
                model.write_bytes(b'test checkpoint, not a pickle')
                sha = handoff.digest(model)
                best = dict(epoch=1, accuracy=100., path=str(model), sha256=sha)
                cm = [[50 if i == j else 0 for j in range(100)] for i in range(100)]
                report = root / 'input.json'
                handoff.write(report, dict(status='complete', spec=case, split_sha256='split',
                    stop_reason='validation patience', best=best,
                    rows=[dict(epoch=i, images=45000, steps=704, skipped_amp_updates=0,
                               train_loss=.1, validation=dict(confusion=cm, accuracy=100., loss=.1))
                          for i in range(1, 11)]))
                handoff.publish(root, case, owner, handoff.digest(plan), dict(best=best, report=str(report),
                                report_sha256=handoff.digest(report)))
            checkpoint = remote / 'export' / cases[1]['id'] / 'best.pth'
            dest = base / 'imported'

            def fake_scp(args, check):
                shutil.copytree(remote / 'export', args[-1])

            with patch.object(handoff.subprocess, 'run', side_effect=fake_scp):
                checkpoint.write_bytes(b'corrupted')
                with self.assertRaisesRegex(RuntimeError, 'mismatched file'):
                    handoff.collect(plan, 'user@host:/results/export', dest)
                self.assertFalse(dest.exists())
                shutil.copyfile(remote / 'input.pth', checkpoint)
                handoff.collect(plan, 'user@host:/results/export', dest)
                index = (dest / 'model_index.csv').read_text()
                self.assertIn('resnet18__p10,pc,100.0', index)
                self.assertIn('resnet18__p20,hpc0,100.0', index)
                self.assertTrue(all((dest / case['id'] / 'best.pth').is_file() for case in cases))
                with self.assertRaisesRegex(RuntimeError, 'will not overwrite'):
                    handoff.collect(plan, 'user@host:/results/export', dest)
