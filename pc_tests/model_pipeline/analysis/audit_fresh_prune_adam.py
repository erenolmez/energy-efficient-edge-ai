"""Audit the fresh-pruning Adam pilot from saved validation and checkpoint evidence."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pc_tests.model_pipeline.paths import PIPELINE

OUT = PIPELINE/'reports/fresh_prune_adam_20261105'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    s=json.loads((OUT/'study.json').read_text())
    assert s['status']=='complete' and not s['test_data_accessed']
    assert s['optimizer']=='Adam' and s['lr']==.001
    for path,h in s['protected_sha256'].items():
        assert digest(path)==h,path
    assert digest(s['best']['path'])==s['best']['sha256']
    assert digest(s['latest']['path'])==s['latest']['sha256']
    assert s['latest']['epoch']==len(s['rows'])
    stop=s['stopping']
    assert stop['min_epochs']<=len(s['rows'])<=stop['max_epochs']
    anchor=-1.
    last_significant=0
    reached=None
    lines=['# Fresh structural pruning followed by Adam recovery — PC pilot','',
           'Original trained 128px ResNet18 baseline, freshly pruned at the requested 30% ratio. '
           'No historical pruned/fine-tuned weights were loaded. Adam LR 0.001, batch 64, AMP, '
           'label smoothing 0.1, no weight decay or scheduler. '
           'Seed 42, original 45,000 training / 5,000 validation split; no test set evaluation.',
           '', '| Epoch | Validation accuracy | Validation loss |',
           '|---:|---:|---:|']
    for epoch,row in enumerate(s['rows'],1):
        assert row['epoch']==epoch and row['images']==45000 and row['steps']==704
        assert 0<=row['skipped_amp_updates']<row['steps']
        v=row['validation']
        cm=np.asarray(v['confusion'])
        assert cm.shape==(100,100) and np.issubdtype(cm.dtype,np.integer)
        assert np.all(cm>=0) and np.all(cm.sum(1)==50) and 100*cm.trace()/5000==v['accuracy']
        assert all(math.isfinite(a) for a in (v['loss'],row['train_loss'],row['train_accuracy']))
        if v['accuracy']>=anchor+stop['min_delta_pp']-1e-9:
            anchor=v['accuracy']
            last_significant=epoch
        if epoch>=stop['min_epochs'] and epoch-last_significant>=stop['patience'] and reached is None:
            reached=epoch
        lines.append(f"| {epoch} | {v['accuracy']:.2f}% | {v['loss']:.3f} |")
    assert (reached==len(s['rows']) and s['stop_reason']=='validation patience') or (
        reached is None and len(s['rows'])==stop['max_epochs'] and s['stop_reason']=='epoch cap')
    best=max(s['rows'],key=lambda row:row['validation']['accuracy'])
    assert best['epoch']==s['best']['epoch'] and best['validation']['accuracy']==s['best']['accuracy']
    d=np.diag(best['validation']['confusion'])
    top=', '.join(s['class_names'][i] for i in np.flatnonzero(d==d.max()))
    bottom=', '.join(s['class_names'][i] for i in np.flatnonzero(d==d.min()))
    tele=[json.loads(line) for line in (OUT/'telemetry.jsonl').read_text().splitlines()]
    assert tele and all('error' not in t and t['temperature_c']<80 for t in tele)
    lines+=['',f"Stopped at epoch {len(s['rows'])}: {s['stop_reason']}. "
             f"Best epoch {best['epoch']}: {s['best']['accuracy']:.2f}% validation. "
             f"Best class(es): {top} ({2*d.max()}%); worst: {bottom} ({2*d.min()}%), 50 images per class.",
             '',f"Freshly pruned start: {s['pruned']['accuracy']:.2f}% validation. "
             f"Actual pruning removed {s['complexity']['parameter_reduction_percent']:.2f}% of parameters "
             f"and {s['complexity']['mac_reduction_percent']:.2f}% of MACs.",
             '', f"CPU audit verified source/checkpoint hashes, validation counts and stopping rule. "
             f"Peak sampled GPU temperature: {max(t['temperature_c'] for t in tele)} °C. "
             'Training timing and energy were not benchmarked.']
    report='\n'.join(lines)+'\n'
    with (OUT/'RESULTS.md').open('x',encoding='utf-8') as f:
        f.write(report)
    print(report)


if __name__=='__main__':
    main()
