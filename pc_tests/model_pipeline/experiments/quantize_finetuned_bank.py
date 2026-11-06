"""Prepare audited QDQ graphs from the new 63-checkpoint bank, CPU only."""
from pathlib import Path
import json
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from pc_tests.model_pipeline.experiments.run_precision_experiments import CalibrationReader, file_hash, write_json
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT
from pc_tests.model_pipeline.runtime.prepare_trt_graph import prepare_graph


def main():
    root = PIPELINE / 'reports/finetuned_routing_20261106'
    graph_dir = root / 'prepared/graphs'
    output = root / 'precision_pool_213'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'quantization.lock').open('x') as lock:
        import os
        lock.write(str(os.getpid()))
    audit = json.loads((REPOSITORY_ROOT / 'progress_reports/2026-11-06/class_validation_audit.json').read_text())
    manifest_path = PIPELINE / 'artifacts/quantization/calibration_indices.json'
    calibration_path = PIPELINE / 'artifacts/pc_precision/data_cache/calibration.npy'
    calibration = np.load(calibration_path, mmap_mode='r', allow_pickle=False)
    manifest = json.loads(manifest_path.read_text())
    assert calibration.shape == (1024, 3, 128, 128) and len(set(manifest['indices'])) == 1024
    assert calibration.dtype == np.float32
    state = dict(status='running', expected_pruned=63, expected_total=213, completed=[],
                 calibration_manifest_sha256=file_hash(manifest_path), calibration_cache_sha256=file_hash(calibration_path),
                 test_data_accessed=False, onnxruntime=ort.__version__, onnx=onnx.__version__)
    baselines = json.loads((PIPELINE / 'reports/gpu_static.json').read_text())['results']
    state['unpruned'] = [dict(name=r['name'], engine=r['engine_path'], sha256=r['engine_sha256'])
                         for r in baselines if r['prune'] == 0]
    assert len({r['name'] for r in state['unpruned']}) == 24
    assert len(audit['models']) == 63
    write_json(output / 'quantization_status.json', state)
    try:
        for name, verified in audit['models'].items():
            state['current'] = name
            source = graph_dir / (name + '.onnx')
            receipt = json.loads((graph_dir / (name + '.json')).read_text())
            assert receipt['checkpoint_sha256'] == verified['checkpoint_sha256']
            assert file_hash(source) == receipt['onnx_sha256']
            folder = output / 'graphs' / name
            folder.mkdir(parents=True, exist_ok=False)
            prepared, qdq, trt_ready = (folder / n for n in ('prepared.onnx', 'qdq.onnx', 'int8_trt.onnx'))
            model = onnx.load(str(source), load_external_data=False)
            input_name = model.graph.input[0].name
            del model
            quant_pre_process(str(source), str(prepared), skip_optimization=True, skip_symbolic_shape=True)
            quantize_static(str(prepared), str(qdq), CalibrationReader(calibration, input_name),
                            quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                            op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
                            calibrate_method=CalibrationMethod.MinMax,
                            calibration_providers=['CPUExecutionProvider'],
                            extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True})
            onnx.checker.check_model(str(qdq))
            model = onnx.load(str(qdq), load_external_data=False)
            assert any(t.data_type == onnx.TensorProto.INT8 for t in model.graph.initializer)
            assert any(n.op_type == 'QuantizeLinear' for n in model.graph.node)
            del model
            folded = prepare_graph(qdq, trt_ready)
            result = dict(name=name, fp_source=str(source), fp_sha256=receipt['onnx_sha256'],
                          checkpoint_sha256=receipt['checkpoint_sha256'], int8_source=str(trt_ready),
                          int8_sha256=file_hash(trt_ready), folded_bias_count=len(folded))
            write_json(folder / 'receipt.json', result)
            state['completed'].append(result)
            write_json(output / 'quantization_status.json', state)
            print(f"QDQ ready {len(state['completed'])}/63: {name}", flush=True)
        state.update(status='complete', current=None)
    except BaseException as exc:
        state.update(status='stopped', error=repr(exc))
        raise
    finally:
        write_json(output / 'quantization_status.json', state)


if __name__ == '__main__':
    main()
