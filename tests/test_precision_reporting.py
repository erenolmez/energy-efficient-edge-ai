import unittest

from pc_tests.model_pipeline.summarize_precision import comparison_rows


class PrecisionReportingTests(unittest.TestCase):
    def test_comparison_uses_measured_p0_and_same_pruning_fp32(self):
        def result(prune, precision, accuracy):
            return dict(architecture='resnet18', prune=prune, precision=precision,
                        test_accuracy=accuracy, onnx_bytes=1048576, test_images=10000,
                        runtime='test', evaluation_seconds=1, status='complete')
        rows = comparison_rows({'results': [result(0, 'fp32', 80),
                                            result(10, 'fp32', 77),
                                            result(10, 'fp16', 76)]})
        half = next(r for r in rows if r['precision'] == 'fp16')
        self.assertEqual(half['loss_vs_p0_pp'], 4)
        self.assertEqual(half['precision_loss_pp'], 1)
        self.assertEqual(half['onnx_mib'], 1)

    def test_incomplete_architecture_is_not_scored_against_another_baseline(self):
        self.assertEqual(comparison_rows({'results': [dict(
            architecture='resnet50', prune=10, precision='int8', status='complete')]}), [])


if __name__ == '__main__':
    unittest.main()
