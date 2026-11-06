"""Reproduce the 0.001 recovery from its original start, then continue to epoch 10."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from pc_tests.model_pipeline.experiments import adam_recovery_curve as curve
from pc_tests.model_pipeline.experiments.compare_dispatch_residency_v2 import digest
from pc_tests.model_pipeline.paths import PIPELINE

PREVIOUS = PIPELINE/'reports/adam_recovery_curve_20261105'
OUT = PIPELINE/'reports/adam_recovery_extension_20261105'
ARTIFACTS = PIPELINE/'artifacts/adam_recovery_extension_20261105'


def audit():
    import torch
    old, new = (json.loads((p/'study.json').read_text()) for p in (PREVIOUS,OUT))
    assert old['status']==new['status']=='complete'
    assert not new['test_data_accessed'] and new['rates']==[.001] and new['epochs']==10
    for key in ('initial_state_sha256','split_sha256','complexity','pruned','seed','batch',
                'optimizer','weight_decay','scheduler','label_smoothing','amp'):
        assert old[key]==new[key], key
    for filename,h in new['protected_sha256'].items():
        assert digest(Path(filename))==h,filename
    replay = {r['epoch']:r for r in old['rows'] if r['lr']==.001}
    assert len(new['rows'])==10 and [r['epoch'] for r in new['rows']]==list(range(1,11))
    all_weight_hashes_match = True
    lines=['# Continuous ten-epoch Adam recovery — ResNet18 p30','',
           'Same freshly pruned weights and original PC CIFAR-100 seed-42 split. '
           'Fresh Adam at 0.001 was used from epoch 1 through epoch 10 without a restart. '
           'The earlier pilot saved deployment weights only, so epochs 1–5 were replayed.',
           '', '| Epoch | Validation accuracy | Validation cross-entropy |',
           '|---:|---:|---:|']
    for r in new['rows']:
        assert r['training_images']==45000 and r['steps']==704
        assert 0<=r['skipped_amp_updates']<704 and r['lr']==.001
        v=r['validation']
        counts=np.asarray(v['confusion'])
        assert counts.shape==(100,100) and np.all(counts>=0) and np.all(counts.sum(1)==50)
        assert v['accuracy']==100*counts.trace()/5000 and math.isfinite(v['loss'])
        assert digest(Path(r['checkpoint']))==r['checkpoint_sha256']
        if r['epoch'] in replay:
            oldrow=replay[r['epoch']]
            assert oldrow['validation']['confusion']==counts.tolist()
            assert digest(Path(oldrow['checkpoint']))==oldrow['checkpoint_sha256']
            a,b = (torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
                   for row in (oldrow,r))
            all_weight_hashes_match &= curve.parameter_hash(a)==curve.parameter_hash(b)
        lines.append(f"| {r['epoch']} | {v['accuracy']:.2f}% | {v['loss']:.3f} |")
    assert all_weight_hashes_match, 'Replay matched predictions but not the saved checkpoint state'
    top=max(new['rows'],key=lambda r:r['validation']['accuracy'])
    counts=np.diag(top['validation']['confusion'])
    names=new['class_names']
    best=', '.join(names[i] for i in np.flatnonzero(counts==counts.max()))
    worst=', '.join(names[i] for i in np.flatnonzero(counts==counts.min()))
    lines+=['','The first five checkpoints reproduced the original state tensors and validation confusion matrices exactly.',
            f"Best observed: epoch {top['epoch']}, {top['validation']['accuracy']:.2f}% validation. "
            f"Best classes: {best} ({2*counts.max()}%); worst classes: {worst} ({2*counts.min()}%).",
            '', 'Validation only; one architecture, pruning level and seed. No new inference latency or energy benchmark.']
    text='\n'.join(lines)+'\n'
    with (OUT/'RESULTS.md').open('x',encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--audit',action='store_true')
    options=parser.parse_args()
    if options.audit:
        audit()
    else:
        if options.data_dir is None:
            parser.error('--data-dir required')
        curve.OUT,curve.MODELS,curve.RATES,curve.EPOCHS=OUT,ARTIFACTS,(.001,),10
        curve.run(options.data_dir)
