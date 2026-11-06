"""CPU audit for the complete 63-case fresh-pruning Adam campaign."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pc_tests.model_pipeline.paths import PIPELINE

OUT = PIPELINE/'reports/fresh_prune_adam_bank_20261105'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    state=json.loads((OUT/'campaign.json').read_text())
    plan=json.loads((OUT/'plan.json').read_text())
    assert state['status']=='complete_pending_cpu_audit' and state['active'] is None
    assert digest(OUT/'plan.json')==state['plan_sha256'] and not plan['test_data_accessed']
    assert len(state['completed'])==len(plan['cases'])==63
    for p,h in plan['protected_sha256'].items():
        assert digest(p)==h,p
    completed={r['id']:r for r in state['completed']}
    assert len(completed)==63 and set(completed)=={r['id'] for r in plan['cases']}
    classes=state['class_names']
    stop=plan['stop']
    results=[]
    for spec in plan['cases']:
        summary=completed[spec['id']]
        assert digest(summary['report'])==summary['report_sha256']
        report=json.loads(Path(summary['report']).read_text())
        assert report['status']=='complete' and report['split_sha256']==plan['split_sha256']
        assert report['best']==summary['best'] and len(report['rows'])==summary['epochs']
        assert digest(report['best']['path'])==report['best']['sha256']
        assert digest(report['latest']['path'])==report['latest']['sha256']
        assert report['latest']['epoch']==summary['epochs']
        reused=bool(summary.get('reused_audited'))
        if reused:
            assert spec['id']==plan['existing_audited_case'] and not report['test_data_accessed']
        else:
            assert report['spec']==spec
        start=report['pruned' if reused else 'pruned_validation']
        assert 0<report['complexity']['pruned_parameters']<report['complexity']['baseline_parameters']
        anchor=-1.
        significant=0
        trigger=None
        for epoch,row in enumerate(report['rows'],1):
            assert row['epoch']==epoch and row['images']==45000
            assert row['steps']==math.ceil(45000/spec['batch'])
            assert 0<=row['skipped_amp_updates']<row['steps']
            cm=np.asarray(row['validation']['confusion'])
            assert cm.shape==(100,100) and np.issubdtype(cm.dtype,np.integer)
            assert np.all(cm>=0) and np.all(cm.sum(1)==50)
            acc=100*cm.trace()/5000
            assert acc==row['validation']['accuracy']
            assert math.isfinite(row['validation']['loss']) and math.isfinite(row['train_loss'])
            if acc>=anchor+stop['min_delta_pp']-1e-9:
                anchor,significant=acc,epoch
            if epoch>=stop['min_epochs'] and epoch-significant>=stop['patience'] and trigger is None:
                trigger=epoch
        assert ((trigger==summary['epochs'] and summary['stop_reason']=='validation patience') or
                (trigger is None and summary['epochs']==stop['max_epochs'] and summary['stop_reason']=='epoch cap'))
        best=max(report['rows'],key=lambda r:r['validation']['accuracy'])
        assert best['epoch']==summary['best']['epoch']
        assert best['validation']['accuracy']==summary['best']['accuracy']
        diag=np.diag(best['validation']['confusion'])
        results.append(dict(id=spec['id'],training_batch=spec['batch'],
            before_adam_validation=start['accuracy'],best_adam_validation=best['validation']['accuracy'],
            best_epoch=best['epoch'],epochs=summary['epochs'],stop_reason=summary['stop_reason'],
            historical_sgd_validation=spec['original_accuracy'],
            parameter_reduction_percent=report['complexity']['parameter_reduction_percent'],
            mac_reduction_percent=report['complexity']['mac_reduction_percent'],
            best_classes=';'.join(classes[i] for i in np.flatnonzero(diag==diag.max())),
            best_class_accuracy=2*int(diag.max()),
            worst_classes=';'.join(classes[i] for i in np.flatnonzero(diag==diag.min())),
            worst_class_accuracy=2*int(diag.min()),checkpoint=summary['best']['path']))
    with (OUT/'RESULTS.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('Audited 63/63 cases. Original inputs and all saved best/latest checkpoints match.')


if __name__=='__main__':
    main()
