import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from pc_tests.model_pipeline.prepare_trt_graph import fold_int32_dequantization


class PrepareTensorRTGraphTests(unittest.TestCase):
    def test_constant_bias_fold_preserves_values(self):
        graph = helper.make_graph(
            [helper.make_node('DequantizeLinear', ['q', 's', 'z'], ['bias'], axis=0, name='bias_dq')],
            'constant', [], [helper.make_tensor_value_info('bias', onnx.TensorProto.FLOAT, [2])],
            initializer=[numpy_helper.from_array(np.array([10, -20], np.int32), 'q'),
                         numpy_helper.from_array(np.array([.1, .2], np.float32), 's'),
                         numpy_helper.from_array(np.array([0, 0], np.int32), 'z')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        result, folded = fold_int32_dequantization(model)
        self.assertEqual(folded, ['bias_dq'])
        self.assertEqual(len(result.graph.node), 0)
        np.testing.assert_allclose(numpy_helper.to_array(result.graph.initializer[0]), [1, -4])

    def test_int8_weights_remain_quantized(self):
        graph = helper.make_graph(
            [helper.make_node('DequantizeLinear', ['q', 's', 'z'], ['weight'])],
            'weights', [], [helper.make_tensor_value_info('weight', onnx.TensorProto.FLOAT, [1])],
            initializer=[numpy_helper.from_array(np.array([10], np.int8), 'q'),
                         numpy_helper.from_array(np.array(.1, np.float32), 's'),
                         numpy_helper.from_array(np.array(0, np.int8), 'z')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        result, folded = fold_int32_dequantization(model)
        self.assertEqual(folded, [])
        self.assertEqual(len(result.graph.node), 1)


if __name__ == '__main__':
    unittest.main()
