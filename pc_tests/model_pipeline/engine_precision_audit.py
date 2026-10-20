"""Identify integer compute paths in TensorRT's detailed layer inspector JSON."""
import json


def integer_compute_layers(details):
    payload = json.loads(details) if isinstance(details, str) else details
    layers = payload.get('Layers', []) if isinstance(payload, dict) else payload
    integer_layers = []
    for layer in layers:
        layer_type = str(layer.get('LayerType', '')).lower()
        if not any(kind in layer_type for kind in ('convolution', 'gemm', 'fullyconnected', 'matrixmultiply')):
            continue
        # Do not count layer names or Q/DQ nodes as proof of integer compute.
        # Require an integer tensor format on a convolution/matrix input.
        formats = [str(tensor.get('Format/Datatype', '')).lower()
                   for tensor in layer.get('Inputs', [])]
        if any('int8' in value for value in formats):
            integer_layers.append({'name': layer.get('Name', ''),
                                   'layer_type': layer.get('LayerType', ''),
                                   'input_formats': formats,
                                   'tactic': layer.get('TacticName', '')})
    return integer_layers
