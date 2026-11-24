"""Run the original plateau protocol at batches 8 and 16 plus a batch-32 anchor."""
import argparse
import json
from pathlib import Path
import time

from pc_tests.model_pipeline.experiments import measure_remaining_plateau as original
from pc_tests.model_pipeline.experiments.remaining_work_design import plateau_windows


def accuracy(output):
    import numpy as np
    import torch
    from pc_tests.model_pipeline.experiments.gpu_static_sweep import accuracy as static_accuracy
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner

    data = original.PIPELINE/'artifacts/pc_precision/data_cache'
    records = json.loads((original.PIPELINE/'reports/gpu_static.json').read_text())['results']
    images = np.load(data/'test.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(data/'test_labels.npy', allow_pickle=False)
    if len(images) != 10000 or len(labels) != 10000:
        raise ValueError('Expected all 10,000 CIFAR-100 test samples')
    results = {}
    with torch.inference_mode():
        for name in ('resnet101__p0__fp16', 'efficientnet_b0__p0__fp16'):
            record = next(r for r in records if r['name'] == name)
            engine = Path(record['engine_path'])
            if original.common.digest(engine) != record['engine_sha256']:
                raise ValueError(f'Historical engine changed: {name}')
            runner = EngineRunner(engine)
            results[name] = {}
            for size in (8,16):
                value, _ = static_accuracy(runner, images, labels, batch_size=size)
                results[name][size] = value
                print(f'{name}, batch {size}: {value:.2f}% test', flush=True)
            if abs(results[name][16] - record['test_accuracy']) > 1e-8:
                raise ValueError(f'Batch-16 accuracy differs from archived score: {name}')
            del runner
    with output.open('x',encoding='utf-8') as stream:
        json.dump(results, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('timing', 'accuracy'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--cool-start-c', type=int, choices=(55,65), default=55)
    args = parser.parse_args()
    if args.mode == 'accuracy':
        accuracy(args.output)
    else:
        original.OUT = args.output.resolve()
        if original.OUT.exists():
            raise FileExistsError('Use a new output directory to preserve prior results')
        original.plateau_windows = lambda: [dict(w, batch={32:8,64:16,128:32}[w['batch']])
                                            for w in plateau_windows()]
        if args.cool_start_c == 65:
            # Keep the historical runner and its source hashes intact. Change
            # only the cooldown threshold and record it before the plan is hashed.
            save = original.atomic_json

            def write_with_gate(path, payload):
                if path == original.OUT/'plan.json':
                    payload['cool_start_max_c'] = 65
                    payload['qualifications'] += ' Alternate <=65C cool-start gate; do not pool with <=55C results.'
                save(path, payload)

            class AlternateGuard(original.CampaignGuard):
                def cool(self):
                    self.phase = 'cooldown'
                    started = time.monotonic()
                    while True:
                        self.check()
                        row = self.reading()
                        elapsed = time.monotonic() - started
                        if elapsed >= 15 and row['temperature_c'] <= 65 and not row['clock_reasons'] & 32:
                            return {'seconds':elapsed, 'reading':row}
                        if elapsed > 180:
                            raise RuntimeError('Cooldown <=65C/no thermal flag not reached in 180s')
                        self.stop.wait(1)

            original.atomic_json = write_with_gate
            original.CampaignGuard = AlternateGuard
        original.main()


if __name__ == '__main__':
    main()
