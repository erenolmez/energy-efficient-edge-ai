"""Evaluate the frozen pruned bank once on the official CIFAR-100 test set."""
import gc
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pc_tests.model_pipeline.analysis.build_class_summary import ARCH, BANK, HERE, LEVELS, REPO, digest, read

import numpy as np
import torch


def main():
    sys.path.insert(0, str(REPO))  # checkpoint class definitions
    audit = read(HERE / 'class_validation_audit.json')
    expected = {f'{a}__p{p}' for a in ARCH for p in LEVELS}
    assert set(audit['models']) == expected
    cache = REPO / 'pc_tests/model_pipeline/artifacts/pc_precision/data_cache'
    image_path, label_path = cache / 'test.npy', cache / 'test_labels.npy'
    images = np.load(image_path, mmap_mode='r', allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    assert images.shape == (10000, 3, 128, 128) and images.dtype == np.float32
    assert labels.shape == (10000,) and np.array_equal(np.bincount(labels), np.full(100, 100))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    batch = 64
    # Hash the exact inputs and freeze the checkpoint list before the sweep.
    provenance = dict(dataset='CIFAR-100 official test', images=10000, examples_per_class=100,
                      image_sha256=digest(image_path), label_sha256=digest(label_path),
                      torch=torch.__version__, dtype='float32', batch=batch,
                      class_names=audit['class_names'])
    data = [(torch.from_numpy(np.array(images[i:i+batch])).contiguous(memory_format=torch.channels_last),
             torch.from_numpy(labels[i:i+batch])) for i in range(0, len(labels), batch)]
    destination = HERE / 'class_test_run.jsonl'
    with destination.open('x', encoding='utf-8') as f, torch.inference_mode():
        print(json.dumps(provenance), file=f, flush=True)
        for index, name in enumerate(audit['models'], 1):
            spec = audit['models'][name]
            path = BANK / name / 'best.pth'
            assert digest(path) == spec['checkpoint_sha256'], name
            model = torch.load(path, map_location='cpu', weights_only=False).eval()
            model = model.to(device='cuda', dtype=torch.float32, memory_format=torch.channels_last)
            assert torch.cuda.mem_get_info()[0] >= 1024**3
            correct = torch.zeros(100, device='cuda', dtype=torch.int64)
            finite = torch.ones((), device='cuda', dtype=torch.bool)
            for x, y in data:
                x, y = x.cuda(), y.cuda()
                logits = model(x)
                assert logits.shape == (len(y), 100) and logits.dtype == torch.float32, name
                finite &= torch.isfinite(logits).all()
                correct.scatter_add_(0, y, (logits.argmax(1) == y).long())
            assert finite.item(), name
            counts = correct.tolist()
            accuracy = sum(counts) / 100
            assert all(0 <= n <= 100 for n in counts)
            print(json.dumps(dict(name=name, checkpoint_sha256=spec['checkpoint_sha256'],
                                  accuracy=accuracy, correct_by_class=counts)), file=f, flush=True)
            print(f'{index}/63 {name}: {accuracy:.2f}% test accuracy', flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
