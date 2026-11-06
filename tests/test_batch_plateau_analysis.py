import unittest
from pc_tests.model_pipeline.analysis.analyze_batch_plateau import percent_change, summarize


class PlateauAnalysisTests(unittest.TestCase):
    def test_direction(self):
        self.assertAlmostEqual(percent_change(1.1, 1), 10)
        self.assertAlmostEqual(percent_change(.9, 1), -10)

    def test_bad_denominator(self):
        with self.assertRaises(ValueError):
            percent_change(1, 0)

    def test_incomplete_not_accepted(self):
        with self.assertRaises(ValueError):
            summarize({'status':'complete','measurements':[]})


if __name__=='__main__':
    unittest.main()
