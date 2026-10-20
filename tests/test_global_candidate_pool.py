import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pc_tests' / 'model_pipeline'))
from global_candidate_pool import EngineCache, global_candidates


class GlobalCandidatesTests(unittest.TestCase):
    def records(self):
        rows = []
        for i in range(213):
            architecture = f'a{i % 8}'
            rows.append(dict(name=f'{architecture}_{i:03}', architecture=architecture,
                             prune=0 if i < 8 else 10, precision='fp32',
                             validation_accuracy=float(i % 8), test_accuracy=100-i,
                             status='complete'))
        return rows

    def test_every_architecture_and_variant_remains_selectable(self):
        records, baseline = global_candidates(self.records())
        self.assertEqual(len(records), 213)
        self.assertEqual(len({r['architecture'] for r in records}), 8)
        self.assertEqual(baseline, 'a7_007')
        for r in records:
            r['test_accuracy'] = -999
        self.assertEqual(global_candidates(records)[1], baseline)

    def test_partial_catalog_rejected(self):
        with self.assertRaises(ValueError):
            global_candidates(self.records()[:-1])

    def test_lru_capacity_does_not_restrict_choices(self):
        records = list(range(213))
        loaded = []
        def factory(record):
            loaded.append(record)
            return ('engine', record)
        cache = EngineCache(records, 2, factory, lambda: None, lambda: None)
        for index in (0, 212, 0, 100):
            self.assertEqual(cache[index], ('engine', index))
            self.assertLessEqual(len(cache.resident), 2)
        self.assertEqual(loaded, [0, 212, 100])
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(len(cache), 213)
        cache.clear()
        self.assertEqual(len(cache.resident), 0)


if __name__ == '__main__':
    unittest.main()
