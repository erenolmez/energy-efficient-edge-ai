import json
from pathlib import Path
import unittest
from unittest.mock import patch
import pc_tests.model_pipeline.experiments.run_p80_batches as study


class P80Tests(unittest.TestCase):
    def setUp(self):
        self.records = [dict(architecture=f'arch{i}', prune=p, precision=q, name=f'arch{i}__p{p}__{q}')
                        for i in range(7) for p in (0,80,90) for q in ('fp32','fp16','int8')]
        self.records += [dict(architecture='shuffle',prune=0,precision=q,name='shuffle_'+q)
                         for q in ('fp32','fp16','int8')]

    def test_exact_p80_not_highest(self):
        chosen = study.select_p80(self.records)
        self.assertEqual(len(chosen),21)
        self.assertTrue(all(r['prune']==80 for r in chosen))

    def test_missing_precision_fails(self):
        records = [r for r in self.records if r['name'] != 'arch0__p80__int8']
        with self.assertRaises(ValueError): study.select_p80(records)

    def test_missing_p80_family_fails(self):
        records = [r for r in self.records if not (r['architecture']=='arch0' and r['prune']==80)]
        with self.assertRaises(ValueError): study.select_p80(records)

    def test_batches_and_count(self):
        self.assertEqual(study.BATCH_SIZES,(8,32,64,128))
        self.assertEqual((21+1)*5*len(study.BATCH_SIZES),440)

    def test_overlap_is_rejected(self):
        with patch.object(study.subprocess,'run') as run:
            run.return_value.stdout = json.dumps(dict(ProcessId=-123,CommandLine='run_random_routes_b64.py'))
            with self.assertRaises(RuntimeError): study.require_no_overlap()

    def test_actual_manifest(self):
        records=json.loads((study.PIPELINE/'reports/gpu_static.json').read_text())['results']
        chosen=study.select_p80(records)
        self.assertEqual(len(chosen),21)
        for record in chosen:
            self.assertTrue(Path(record['engine_path']).is_file())
        self.assertNotIn('most_pruned_b128',str(study.REPORT))


if __name__=='__main__': unittest.main()
