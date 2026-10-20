"""Fold unsupported constant INT32 dequantization without recalibration."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def fold_int32_dequantization(model):
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    retained, replacements, folded = [], [], []
    for node in model.graph.node:
        if (node.op_type != 'DequantizeLinear' or len(node.input) < 2
                or node.input[0] not in constants or node.input[1] not in constants
                or constants[node.input[0]].dtype != np.int32
                or (len(node.input) > 2 and node.input[2] not in constants)):
            retained.append(node)
            continue
        quantized = constants[node.input[0]]
        scale = constants[node.input[1]]
        zero = constants[node.input[2]] if len(node.input) > 2 else np.array(0, dtype=np.int32)
        axis = next((attribute.i for attribute in node.attribute if attribute.name == 'axis'), 1)
        if scale.ndim == 1:
            shape = [1] * quantized.ndim
            shape[axis] = len(scale)
            scale = scale.reshape(shape)
            if zero.ndim == 1:
                zero = zero.reshape(shape)
        value = (quantized.astype(np.float32) - zero.astype(np.float32)) * scale.astype(np.float32)
        replacements.append(numpy_helper.from_array(value.astype(np.float32), node.output[0]))
        folded.append(node.name)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    replaced_names = {tensor.name for tensor in replacements}
    originals = [t for t in model.graph.initializer if t.name not in replaced_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(originals + replacements)
    referenced = {name for node in retained for name in node.input}
    referenced.update(value.name for value in model.graph.output)
    referenced.update(value.name for value in model.graph.input)
    active = [t for t in model.graph.initializer if t.name in referenced]
    del model.graph.initializer[:]
    model.graph.initializer.extend(active)
    onnx.checker.check_model(model)
    return model, folded


def prepare_graph(source, destination):
    model, folded = fold_int32_dequantization(onnx.load(str(source)))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return folded
