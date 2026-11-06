"""CPU-only verification and report for the matched Adam pilot."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pc_tests.model_pipeline.paths import PIPELINE

OUT = PIPELINE/'reports/adam_recovery_curve_20261105'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    s = json.loads((OUT/'study.json').read_text())
    assert s['status']=='complete' and s['test_data_accessed'] is False
    assert s['epochs']==5 and s['rates']==[.01,.001] and s['optimizer']=='Adam'
    assert len(s['initial_state_sha256'])==64 and s['pruned']['accuracy']==1.
    for path,h in s['protected_sha256'].items():
        assert digest(path)==h,path
    rows = {(r['lr'],r['epoch']):r for r in s['rows']}
    assert len(rows)==len(s['rows'])==10
    assert set(rows)=={(lr,e) for lr in s['rates'] for e in range(1,6)}
    for r in rows.values():
        assert r['training_images']==45000 and r['steps']==704
        assert 0<=r['skipped_amp_updates']<r['steps']
        assert digest(r['checkpoint'])==r['checkpoint_sha256']
        a=np.asarray(r['validation']['confusion'])
        assert a.shape==(100,100) and np.issubdtype(a.dtype,np.integer)
        assert np.all(a>=0) and np.all(a.sum(1)==50)
        assert 100*a.trace()/5000==r['validation']['accuracy']
        assert all(math.isfinite(v) and v>=0 for v in (r['train_loss'],r['validation']['loss']))
    t=[json.loads(l) for l in (OUT/'telemetry.jsonl').read_text().splitlines()]
    assert t and all('error' not in r and r['temperature_c']<80 for r in t)
    lines=['# ResNet18 p30: one to five Adam epochs','',
           'Same freshly pruned 128px PC checkpoint and seed-42 training/validation split, separate fresh Adam optimizers. '
           'Each epoch processed all 45,000 training images; validation used the same 5,000 images. No test set evaluation.',
           '', '| Epoch | Adam 0.01 validation accuracy | Adam 0.001 validation accuracy | Adam 0.01 validation loss | Adam 0.001 validation loss |',
           '|---:|---:|---:|---:|---:|']
    for e in range(1,6):
        a,b=(rows[lr,e]['validation'] for lr in s['rates'])
        lines.append(f"| {e} | {a['accuracy']:.2f}% | {b['accuracy']:.2f}% | {a['loss']:.3f} | {b['loss']:.3f} |")
    lines += ['',f"Freshly pruned: {s['pruned']['accuracy']:.2f}% validation. Historical best across 20 SGD epochs: "
              f"{s['historical_20_epoch_sgd_validation_accuracy']:.2f}% validation (reference only, not a matched optimizer comparison).",
              '', 'Best and worst classes for the best observed epoch of each rate (50 images per class):','']
    for lr in s['rates']:
        r=max((rows[lr,e] for e in range(1,6)),key=lambda row:row['validation']['accuracy'])
        d=np.diag(r['validation']['confusion'])
        names=s['class_names']
        top=', '.join(names[i] for i in np.flatnonzero(d==d.max()))
        bottom=', '.join(names[i] for i in np.flatnonzero(d==d.min()))
        lines.append(f"- LR {lr:g}, epoch {r['epoch']}: {r['validation']['accuracy']:.2f}% validation. "
                     f"Best: {top} ({2*d.max()}%); worst: {bottom} ({2*d.min()}%).")
    peak=max(r['temperature_c'] for r in t)
    flags=sum(bool(r['clock_reasons']&32) for r in t)
    lines += ['',f"CPU audit verified 10 checkpoints and confusion matrices. Sampled peak {peak} °C; "
              f"{flags} sampled software thermal flags. Training timing/energy was not a benchmark.",
              '', 'Single model, pruning level and seed. The validation curve does not establish convergence or generalization to the HPC sweep.']
    report='\n'.join(lines)+'\n'
    with (OUT/'RESULTS.md').open('x',encoding='utf-8') as f:
        f.write(report)
    print(report)


if __name__=='__main__':
    main()
