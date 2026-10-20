import unittest

from pc_tests.model_pipeline.engine_precision_audit import integer_compute_layers


class EnginePrecisionAuditTests(unittest.TestCase):
    def test_quantized_convolution_is_detected(self):
        layers = {'Layers': [{'Name': 'conv', 'LayerType': 'CaskConvolution',
                              'Inputs': [{'Format/Datatype': 'Row major Int8 format'}]}]}
        self.assertEqual(len(integer_compute_layers(layers)), 1)

    def test_names_and_quantization_nodes_are_not_integer_compute_proof(self):
        layers = {'Layers': [
            {'Name': 'int8_conv', 'LayerType': 'CaskConvolution',
             'Inputs': [{'Format/Datatype': 'Float format'}]},
            {'Name': 'quantize', 'LayerType': 'Quantize',
             'Inputs': [{'Format/Datatype': 'Int8 format'}]},
        ]}
        self.assertEqual(integer_compute_layers(layers), [])


if __name__ == '__main__':
    unittest.main()
