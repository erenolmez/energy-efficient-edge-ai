"""Evaluate eight frozen single-run validation leads on untouched-for-selection CIFAR-100 test."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from torchvision import datasets, transforms

from hpc.full_factorial_protocol import build_model, make_plan

CASES = ((139,48),(329,48),(314,42),(312,42),(415,42),(1678,48),(237,48),(228,45))
PLAN_HASH = '9b3173c990036c44f2a1b7b09d1d38000a82cf2c03f796ff4d58047b8153f50c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('pc_static', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the PC comparison')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    old = json.loads(args.pc_static.read_text(encoding='utf-8'))
    with tarfile.open(args.archive, 'r:') as archive:
        def member(name):
            entry = archive.getmember(name)
            if not entry.isfile():
                raise ValueError(f'Not a regular archive member: {name}')
            return archive.extractfile(entry)

        with member('plan.json') as stream:
            plan = json.load(stream)
        with member('data_ready.json') as stream:
            receipt = json.load(stream)
        if plan != make_plan() or plan['sha256'] != PLAN_HASH or receipt['plan_sha256'] != PLAN_HASH:
            raise ValueError('Frozen plan mismatch')
        verified = []
        for cid, seed in CASES:
            trial = plan['trials'][cid]
            prefix = f'case_{cid:04d}/seed_{seed}'
            with member(f'{prefix}/summary.json') as stream:
                summary = json.load(stream)
            if (trial['seed'] != seed or summary['trial'] != trial or summary['seed'] != seed
                    or summary['plan_sha256'] != PLAN_HASH or summary['split_sha256'] != receipt['split_sha256']
                    or summary['status'] != 'complete' or summary['epochs_complete'] != 200):
                raise ValueError(f'Unexpected trainer summary: {prefix}')
            with member(f'{prefix}/latest.pt') as stream:
                sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            if sha256 != summary['checkpoint_sha256']:
                raise ValueError(f'Checkpoint digest mismatch: {prefix}')
            baseline = next(r for r in old['results'] if r['name'] == trial['architecture']+'__p0__fp32')
            if baseline['test_images'] != 10000 or baseline['validation_images'] != 5000:
                raise ValueError('Historical split counts changed')
            verified.append((prefix, trial, summary, baseline))

        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(plan['normalization_mean'], plan['normalization_std'])])
        test = datasets.CIFAR100(args.data, train=False, transform=transform, download=False)
        labels = np.load(args.pc_static.parent.parent/'artifacts/pc_precision/data_cache/test_labels.npy', allow_pickle=False)
        if len(test) != 10000 or not np.array_equal(test.targets, labels):
            raise ValueError('Test size or order differs from historical PC cache')
        loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        rows = []
        with torch.inference_mode():
            for prefix, trial, summary, baseline in verified:
                # Trusted training output, hash-verified against the trainer's separate summary.
                with member(f'{prefix}/latest.pt') as stream, torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    checkpoint = torch.load(stream, weights_only=True, map_location='cpu')
                if (checkpoint['epoch'] != 200 or checkpoint['recipe']['trial'] != trial
                        or checkpoint['split_sha256'] != receipt['split_sha256']
                        or checkpoint['best_accuracy'] != summary['best_validation_accuracy']
                        or not checkpoint['best_state']):
                    raise ValueError(f'Checkpoint metadata mismatch: {prefix}')
                model = build_model(trial['architecture'])
                model.load_state_dict(checkpoint['best_state'], strict=True)
                del checkpoint
                model = model.cuda().eval()
                correct = 0
                for inputs, targets in loader:
                    logits = model(inputs.cuda(non_blocking=True))
                    if not logits.isfinite().all().item():
                        raise ValueError(f'Nonfinite test output: {prefix}')
                    correct += logits.argmax(1).cpu().eq(targets).sum().item()
                row = dict(trial=trial, checkpoint_sha256=summary['checkpoint_sha256'],
                    hpc_best_validation=summary['best_validation_accuracy'], hpc_test=correct/100,
                    pc_validation=baseline['validation_accuracy'], pc_test=baseline['test_accuracy'])
                rows.append(row)
                print(json.dumps(row), flush=True)
                del model
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump({'plan_sha256': PLAN_HASH, 'test_images': 10000,
                'scratch_inference': 'PyTorch FP32, native 32px, local PC GPU',
                'historical_inference': 'unpruned TensorRT FP32, 128px',
                'selection': 'frozen single-run validation leads; descriptive, not three-seed winners',
                'rows': rows}, stream, indent=2)


if __name__ == '__main__':
    main()
