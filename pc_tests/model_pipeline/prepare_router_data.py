"""Prepare validation supervision and shared RGB inputs without test-label fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_pipeline.data import CIFAR100_MEAN, CIFAR100_STD, evaluation_transform
from pc_tests.model_pipeline.run_precision_experiments import write_json


def rgb_from_normalized(images):
    mean = np.asarray(CIFAR100_MEAN, np.float32)[None, :, None, None]
    std = np.asarray(CIFAR100_STD, np.float32)[None, :, None, None]
    return np.clip(np.rint((images * std + mean) * 255), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    split = json.loads(args.split_manifest.read_text())
    indices = split['validation_indices']
    if len(indices) != 5000 or set(indices) & set(split['train_indices']):
        raise ValueError('Invalid validation split')
    fingerprint = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / 'router_data.json'
    if marker.exists() and json.loads(marker.read_text()).get('validation_indices_sha256') == fingerprint:
        print('Router validation cache already complete')
        return
    dataset = CIFAR100(args.data_dir, train=True, download=False, transform=evaluation_transform(128))
    validation = np.lib.format.open_memmap(cache / 'validation.npy', mode='w+', dtype=np.float32,
                                         shape=(5000,3,128,128))
    labels = []
    for position, index in enumerate(indices):
        image, label = dataset[index]
        validation[position] = image.numpy()
        labels.append(label)
    validation.flush()
    del validation
    np.save(cache / 'validation_labels.npy', np.asarray(labels, np.int64))
    for name, count in (('test',10000), ('validation',5000)):
        normalized = np.load(cache / f'{name}.npy', mmap_mode='r')
        if normalized.shape != (count,3,128,128):
            raise ValueError('Unexpected classifier cache shape')
        rgb = np.lib.format.open_memmap(cache / f'{name}_rgb128.npy', mode='w+', dtype=np.uint8,
                                       shape=(count,3,128,128))
        for start in range(0, count, 128):
            rgb[start:start+128] = rgb_from_normalized(normalized[start:start+128])
        rgb.flush()
        del rgb, normalized
    write_json(marker, {'validation_indices': indices, 'validation_indices_sha256': fingerprint,
                        'split_sha256': split['sha256'], 'validation_images': 5000,
                        'test_images': 10000, 'rgb_input': 'PIL-resized 128x128 RGB uint8; normalization is performed on GPU in the complete routed/static paths',
                        'test_role': 'held-out evaluation only; not router fitting or calibration'})
    print('Prepared validation and common RGB128 inputs')


if __name__ == '__main__':
    main()
