"""Identify integer compute paths in TensorRT's detailed layer inspector JSON."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

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
