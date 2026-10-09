"""Check PyTorch/ONNX numerical parity on fixed CIFAR-100 images."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_pipeline.data import evaluation_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', default='1,8,128')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    batches = [int(x) for x in args.batches.split(',')]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = json.loads(args.manifest.read_text())['models']
    dataset = CIFAR100(args.data_dir, train=False, download=False,
                       transform=evaluation_transform(128))
    indices = np.random.default_rng(42).choice(len(dataset), max(batches), replace=False)
    inputs = torch.stack([dataset[int(i)][0] for i in indices])
    input_hash = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for source in sources:
        record = {'source_stem': source['source_stem'], 'checks': []}
        model = session = None
        try:
            model = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
            model.float().eval().to(args.device)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(source['onnx'], options,
                                           providers=['CPUExecutionProvider'])
            for batch in batches:
                x = inputs[:batch]
                with torch.inference_mode():
                    # Keep GPU memory bounded while checking the ONNX runtime's full batch.
                    expected = torch.cat([model(chunk.to(args.device)).cpu()
                                          for chunk in x.split(8)]).numpy()
                actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
                finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
                shape_ok = actual.shape == expected.shape == (batch, 100)
                close = bool(shape_ok and finite and np.allclose(
                    expected, actual, atol=1e-3, rtol=1e-3))
                record['checks'].append({
                    'batch': batch, 'shape': list(actual.shape), 'finite': finite,
                    'allclose': close, 'atol': 1e-3, 'rtol': 1e-3,
                    'max_abs_error': float(np.max(np.abs(expected - actual))),
                    'mean_abs_error': float(np.mean(np.abs(expected - actual))),
                    'top1_agreement_percent': float(100 * np.mean(
                        expected.argmax(1) == actual.argmax(1))),
                })
            record['status'] = 'passed' if all(c['allclose'] for c in record['checks']) else 'failed'
        except Exception as exc:
            record['status'] = 'error'
            record['error'] = str(exc)
        finally:
            del model, session
            gc.collect()
            torch.cuda.empty_cache()
        results.append(record)
        report = {
            'scope': 'FP32 export equivalence; not full test-set accuracy or quantization evaluation',
            'seed': 42, 'image_indices': indices.tolist(), 'input_sha256': input_hash,
            'batches': batches, 'pytorch_device': args.device,
            'onnx_provider': 'CPUExecutionProvider',
            'torch_version': torch.__version__, 'onnxruntime_version': ort.__version__,
            'seconds': time.perf_counter() - started, 'expected_models': len(sources),
            'results': results,
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f"[{len(results)}/{len(sources)}] {source['source_stem']}: {record['status']}", flush=True)
    if any(r['status'] != 'passed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
