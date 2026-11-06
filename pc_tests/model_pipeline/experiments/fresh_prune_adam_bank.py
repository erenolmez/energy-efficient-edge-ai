"""Sequential PC campaign for all 63 original structural-pruning combinations."""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from torch.utils.data import Subset

from model_pipeline.data import evaluation_transform, training_transform, split_fingerprint, stratified_indices
from model_pipeline.reproducibility import seed_everything
from pc_tests.model_pipeline.experiments import fresh_prune_adam as pilot
from pc_tests.model_pipeline.experiments.compare_dispatch_residency_v2 import Guard, digest, utc
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as telemetry
from pc_tests.model_pipeline.experiments.prune_finetune import make_loader, structurally_prune
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE/'reports/fresh_prune_adam_bank_20261105'
MODELS = PIPELINE/'artifacts/fresh_prune_adam_bank_20261105'
BASE = PIPELINE/'artifacts/baselines'
PRUNED = PIPELINE/'artifacts/pruned'
ARCH = ('resnet18','resnet34','resnet50','resnet101','efficientnet_b0',
        'mobilenetv3_small','mobilenetv3_large')
LEVELS = tuple(range(10,100,10))
BATCH = {'resnet18':64, 'resnet34':32, 'resnet50':16, 'resnet101':16,
         'efficientnet_b0':32, 'mobilenetv3_small':64, 'mobilenetv3_large':32}
PILOT_ID = 'resnet18__p30'


def freeze():
    """Only the seven architectures and nine levels in the original bank."""
    old_report=pilot.OUT/'study.json'
    old=json.loads(old_report.read_text())
    assert (pilot.OUT/'RESULTS.md').is_file()
    assert old['status']=='complete' and old['architecture']=='resnet18' and old['pruning_ratio']==.3
    assert old['optimizer']=='Adam' and old['lr']==pilot.LR and old['batch']==BATCH['resnet18']
    assert old['stopping']==dict(min_epochs=pilot.MIN_EPOCHS,patience=pilot.PATIENCE,
                                max_epochs=pilot.MAX_EPOCHS,min_delta_pp=pilot.MIN_DELTA_PP)
    assert digest(Path(old['best']['path']))==old['best']['sha256']
    manifests={}
    rows=[]
    split_path=BASE/'cifar100_split_seed42.json'
    split=json.loads(split_path.read_text())
    for arch in ARCH:
        baseline=BASE/f'{arch}__p0__fp32.pth'
        baseline_meta=baseline.with_suffix('.json')
        meta=json.loads(baseline_meta.read_text())
        assert meta['architecture']==arch and meta['split_sha256']==split['sha256']
        assert meta['image_size']==128 and not meta['smoke_test'] and meta['prune']==0
        manifests[str(baseline)]=digest(baseline)
        manifests[str(baseline_meta)]=digest(baseline_meta)
        for level in LEVELS:
            name=f'{arch}__p{level}'
            prior=PRUNED/f'{name}__fp32.json'
            assert prior.with_suffix('.pth').is_file()
            previous=json.loads(prior.read_text())
            assert (previous['architecture'],previous['requested_prune_percent'])==(arch,level)
            assert previous['split_sha256']==split['sha256']
            manifests[str(prior)]=digest(prior)
            rows.append(dict(id=name,architecture=arch,level=level,batch=BATCH[arch],
                baseline=str(baseline),original_sidecar=str(prior),original_accuracy=previous['best_validation_accuracy']))
    for p in (Path(__file__),split_path,old_report,pilot.OUT/'RESULTS.md',
              PIPELINE/'experiments/prune_finetune.py',PIPELINE/'experiments/fresh_prune_adam.py',
              PIPELINE/'experiments/adam_recovery_curve.py'):
        manifests[str(p)]=digest(p)
    assert len(rows)==63 and len({r['id'] for r in rows})==63
    return dict(cases=rows, protected_sha256=manifests, split_sha256=split['sha256'],
                optimizer='Adam', lr=pilot.LR, seed=42, image_size=128, amp=True,
                label_smoothing=.1, weight_decay=0, scheduler=None,
                stop=dict(min_epochs=pilot.MIN_EPOCHS,patience=pilot.PATIENCE,
                          max_epochs=pilot.MAX_EPOCHS,min_delta_pp=pilot.MIN_DELTA_PP),
                existing_audited_case=PILOT_ID, test_data_accessed=False),split,old


def run_case(spec, plan, train, val_loader, guard, check, pause_heat):
    from pc_tests.model_pipeline.experiments.adam_recovery_curve import parameter_hash, validate
    name,batch=spec['id'],spec['batch']
    report=OUT/'cases'/name/'study.json'
    output=MODELS/name
    output.mkdir(parents=True,exist_ok=False)
    state=dict(status='preparing',spec=spec,rows=[],split_sha256=plan['split_sha256'],
               started_utc=utc())

    def save():
        atomic_json(report,state)

    try:
        seed_everything(42)
        pause_heat()
        guard.phase=f'{name}:pruning'
        p=Path(spec['baseline'])
        assert digest(p)==plan['protected_sha256'][str(p)]
        model=torch.load(p,map_location='cpu',weights_only=False).cuda().eval()
        args=SimpleNamespace(image_size=128,round_to=8,global_pruning=True,isomorphic=True,max_pruning_ratio=.95)
        state['complexity']=structurally_prune(model,spec['level']/100,args,torch.device('cuda'))
        complexity=state['complexity']
        if not 0<complexity['pruned_parameters']<complexity['baseline_parameters']:
            raise RuntimeError('Expected real structural parameter removal')
        # A different retained topology is recorded rather than silently equated.
        old=json.loads(Path(spec['original_sidecar']).read_text())
        state['matches_historical_complexity']=all(complexity[k]==old[k] for k in
            ('baseline_parameters','pruned_parameters','baseline_macs','pruned_macs'))
        state['initial_state_sha256']=parameter_hash(model)
        state['pruned_validation']=validate(model,val_loader,check)
        seed_everything(42)
        loader=make_loader(train,batch,2,True,42)
        opt=torch.optim.Adam(model.parameters(),lr=pilot.LR)
        scaler=torch.amp.GradScaler('cuda')
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=.1)
        best=anchor=-1.
        last_significant=0
        state['status']='training'
        save()
        for epoch in range(1,pilot.MAX_EPOCHS+1):
            guard.phase=f'{name}:train:{epoch}'
            model.train()
            total=correct=skipped=0
            total_loss=0.
            for step,(x,y) in enumerate(loader,1):
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    z=model(x)
                    loss=loss_fn(z,y)
                value=float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite training loss')
                scale=scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                skipped+=scaler.get_scale()<scale
                total+=len(y)
                correct+=int((z.argmax(1)==y).sum())
                total_loss+=value*len(y)
                check()
                if step%8==0:
                    pause_heat()
            assert total==45000 and step==math.ceil(45000/batch) and skipped<step
            guard.phase=f'{name}:validation:{epoch}'
            v=validate(model,val_loader,check)
            acc=v['accuracy']
            if acc>best:
                best=acc
                f=output/f'best_epoch{epoch:02}.pth'
                with f.open('xb') as handle:
                    torch.save(copy.deepcopy(model).cpu().eval(),handle)
                state['best']=dict(epoch=epoch,accuracy=acc,path=str(f),sha256=digest(f))
            if acc>=anchor+pilot.MIN_DELTA_PP-1e-9:
                anchor,last_significant=acc,epoch
            # New case directory only; preserve the latest optimizer/model pair atomically.
            latest=output/'latest.pt'
            pending=output/'latest.pending'
            with pending.open('xb') as f:
                torch.save(dict(model=copy.deepcopy(model).cpu().eval(),optimizer=opt.state_dict(),
                                scaler=scaler.state_dict(),epoch=epoch,seed=42,
                                initial_state_sha256=state['initial_state_sha256']),f)
            os.replace(pending,latest)
            state['latest']=dict(epoch=epoch,path=str(latest),sha256=digest(latest))
            state['rows'].append(dict(epoch=epoch,images=total,steps=step,skipped_amp_updates=skipped,
                train_accuracy=100*correct/total,train_loss=total_loss/total,validation=v))
            save()
            print(f'{name} epoch {epoch}: validation {acc:.2f}%, best {best:.2f}%',flush=True)
            if epoch>=pilot.MIN_EPOCHS and epoch-last_significant>=pilot.PATIENCE:
                state['stop_reason']='validation patience'
                break
        else:
            state['stop_reason']='epoch cap'
        assert digest(Path(state['best']['path']))==state['best']['sha256']
        state.update(status='complete',finished_utc=utc())
        save()
        return dict(id=name,epochs=len(state['rows']),best=state['best'],
                    historical_sgd_accuracy=spec['original_accuracy'],
                    stop_reason=state['stop_reason'],report=str(report),report_sha256=digest(report))
    except BaseException as exc:
        state.update(status='stopped',error=repr(exc),finished_utc=utc())
        save()
        raise


def run(data_dir):
    from pc_tests.model_pipeline.experiments.adam_post_prune_pilot import TrainOnlyCIFAR100
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as lock:
        json.dump(dict(pid=os.getpid(),started_utc=utc()),lock)
    telemetry.OUT=OUT
    guard=Guard()
    state=dict(status='preparing',expected=63,completed=[],active=None)

    def save():
        atomic_json(OUT/'campaign.json',state)

    def check():
        guard.check()
        if torch.cuda.mem_get_info()[0]<1024**3:
            raise RuntimeError('Below 1 GiB VRAM reserve')

    def pause_heat():
        check()
        if guard.last['temperature_c']<76:
            return
        torch.cuda.synchronize()
        phase=guard.phase
        guard.phase='thermal_cooldown'
        deadline=time.monotonic()+180
        while guard.last['temperature_c']>68:
            check()
            if time.monotonic()>deadline:
                raise RuntimeError('Cooldown did not reach 68C in 180 seconds')
            time.sleep(.5)
        guard.phase=phase

    try:
        state['initial_telemetry']=guard.start()
        torch.set_num_threads(1)
        plan,split,old=freeze()
        atomic_json(OUT/'plan.json',plan)
        state['plan_sha256']=digest(OUT/'plan.json')
        data=TrainOnlyCIFAR100(root=data_dir,train=True,download=False,transform=training_transform(128))
        evaluation=copy.copy(data)
        evaluation.transform=evaluation_transform(128)
        train,val=stratified_indices(data.targets,5000,42)
        assert train==split['train_indices'] and val==split['validation_indices']
        assert split_fingerprint(train,val)==plan['split_sha256']
        val_loader=make_loader(Subset(evaluation,val),32,2,False,43)
        training=Subset(data,train)
        p=pilot.OUT/'study.json'
        state['completed'].append(dict(id=PILOT_ID,epochs=len(old['rows']),best=old['best'],
            historical_sgd_accuracy=next(r['original_accuracy'] for r in plan['cases'] if r['id']==PILOT_ID),
            stop_reason=old['stop_reason'],report=str(p),report_sha256=digest(p),reused_audited=True))
        state.update(status='running',gpu=torch.cuda.get_device_name(0),class_names=data.classes)
        save()
        for spec in plan['cases']:
            if spec['id']==PILOT_ID:
                continue
            state['active']=spec['id']
            save()
            state['completed'].append(run_case(spec,plan,training,val_loader,guard,check,pause_heat))
            state['active']=None
            save()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Completed {len(state['completed'])}/63: {spec['id']}",flush=True)
        assert len(state['completed'])==63 and digest(OUT/'plan.json')==state['plan_sha256']
        assert all(digest(Path(p))==h for p,h in plan['protected_sha256'].items())
        state.update(status='complete_pending_cpu_audit',finished_utc=utc())
        save()
    except BaseException as exc:
        state.update(status='stopped',error=repr(exc),finished_utc=utc())
        save()
        raise
    finally:
        guard.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    run(parser.parse_args().data_dir)
