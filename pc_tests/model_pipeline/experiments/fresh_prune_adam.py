"""Fresh structural pruning of a historical unpruned checkpoint, then Adam recovery."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Subset

from model_pipeline.data import evaluation_transform, training_transform, split_fingerprint, stratified_indices
from model_pipeline.reproducibility import seed_everything
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as common
from pc_tests.model_pipeline.experiments.adam_post_prune_pilot import TrainOnlyCIFAR100
from pc_tests.model_pipeline.experiments.adam_recovery_curve import parameter_hash, validate
from pc_tests.model_pipeline.experiments.prune_finetune import make_loader, structurally_prune
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE/'reports/fresh_prune_adam_20261105'
MODELS = PIPELINE/'artifacts/fresh_prune_adam_20261105'
LR = .001
MIN_EPOCHS, PATIENCE, MAX_EPOCHS = 10, 5, 25
MIN_DELTA_PP = .20


def run(data_dir):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f:
        json.dump(dict(pid=os.getpid(),utc=common.utc()),f)
    common.OUT=OUT
    guard=common.Guard()
    state=dict(status='preparing',architecture='resnet18',pruning_ratio=.30,
               optimizer='Adam',lr=LR,seed=42,batch=64,image_size=128,amp=True,
               weight_decay=0,label_smoothing=.1,test_data_accessed=False,rows=[],
               stopping=dict(min_epochs=MIN_EPOCHS,patience=PATIENCE,max_epochs=MAX_EPOCHS,
                             min_delta_pp=MIN_DELTA_PP))

    def save():
        atomic_json(OUT/'study.json',state)

    def check():
        guard.check()
        if torch.cuda.mem_get_info()[0] < 1024**3:
            raise RuntimeError('Below 1 GiB VRAM reserve')

    try:
        state['start_telemetry']=guard.start()
        torch.set_num_threads(1)
        seed_everything(42)
        baseline=PIPELINE/'artifacts/baselines/resnet18__p0__fp32.pth'
        split_path=baseline.parent/'cifar100_split_seed42.json'
        old=PIPELINE/'artifacts/pruned/resnet18__p30__fp32.pth'
        curve_path=PIPELINE/'reports/adam_recovery_curve_20261105/study.json'
        state['protected_sha256']={str(p):common.digest(p) for p in (
            baseline,baseline.with_suffix('.json'),split_path,old,old.with_suffix('.json'),curve_path,
            Path(__file__),PIPELINE/'experiments/prune_finetune.py',
            PIPELINE/'experiments/adam_recovery_curve.py')}
        previous,split=(json.loads(p.read_text()) for p in (curve_path,split_path))
        assert previous['status']=='complete'
        assert state['protected_sha256'][str(baseline)]==previous['protected_sha256'][str(baseline)]
        ds=TrainOnlyCIFAR100(root=data_dir,train=True,download=False,transform=training_transform(128))
        eval_ds=copy.copy(ds)
        eval_ds.transform=evaluation_transform(128)
        train,val=stratified_indices(ds.targets,5000,42)
        assert train==split['train_indices'] and val==split['validation_indices']
        assert split_fingerprint(train,val)==split['sha256']==previous['split_sha256']
        state.update(split_sha256=split['sha256'],class_names=ds.classes,gpu=torch.cuda.get_device_name(0))
        val_loader=make_loader(Subset(eval_ds,val),64,2,False,43)
        guard.phase='fresh_pruning'
        model=torch.load(baseline,map_location='cpu',weights_only=False).cuda().eval()
        config=SimpleNamespace(image_size=128,round_to=8,global_pruning=True,isomorphic=True,max_pruning_ratio=.95)
        state['complexity']=structurally_prune(model,.30,config,torch.device('cuda'))
        assert state['complexity']==previous['complexity']
        state['pruned']=validate(model,val_loader,check)
        state['initial_state_sha256']=parameter_hash(model)
        assert state['initial_state_sha256']==previous['initial_state_sha256']
        assert state['pruned']['confusion']==previous['pruned']['confusion']
        seed_everything(42)
        train_loader=make_loader(Subset(ds,train),64,2,True,42)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        scaler=torch.amp.GradScaler('cuda')
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=.1)
        MODELS.mkdir(parents=True,exist_ok=True)
        best=anchor=-1.
        significant_epoch=0
        state['status']='running'
        save()
        print('Fresh p30 checkpoint verified; starting Adam at 0.001',flush=True)
        for epoch in range(1,MAX_EPOCHS+1):
            guard.phase=f'train_epoch_{epoch}'
            model.train()
            total=correct=skipped=0
            summed_loss=0.
            for step,(x,y) in enumerate(train_loader,1):
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
                summed_loss+=value*len(y)
                check()
            assert total==45000 and step==704
            guard.phase=f'validate_epoch_{epoch}'
            result=validate(model,val_loader,check)
            acc=result['accuracy']
            if acc>best:
                best=acc
                file=MODELS/f'best_epoch{epoch:02}.pth'
                with file.open('xb') as f:
                    torch.save(copy.deepcopy(model).cpu().eval(),f)
                state['best']=dict(epoch=epoch,accuracy=acc,path=str(file),sha256=common.digest(file))
            if acc>=anchor+MIN_DELTA_PP-1e-9:
                anchor=acc
                significant_epoch=epoch
            # Keep optimizer/scaler state and model weights together. Persistent
            # loader worker RNG is not serialized; a future restart need not be bit-exact.
            latest=MODELS/'latest.pt'
            tmp=MODELS/'latest.pending'
            with tmp.open('xb') as f:
                torch.save(dict(model=copy.deepcopy(model).cpu().eval(),optimizer=opt.state_dict(),
                                scaler=scaler.state_dict(),epoch=epoch,seed=42,
                                initial_state_sha256=state['initial_state_sha256']),f)
            os.replace(tmp,latest)
            state['latest']=dict(epoch=epoch,path=str(latest),sha256=common.digest(latest))
            state['rows'].append(dict(epoch=epoch,images=total,steps=step,
                skipped_amp_updates=skipped,train_accuracy=100*correct/total,
                train_loss=summed_loss/total,validation=result))
            save()
            print(f'Epoch {epoch}: validation {acc:.2f}%, best {best:.2f}%',flush=True)
            if epoch>=MIN_EPOCHS and epoch-significant_epoch>=PATIENCE:
                state['stop_reason']='validation patience'
                break
        else:
            state['stop_reason']='epoch cap'
        assert all(common.digest(Path(p))==h for p,h in state['protected_sha256'].items())
        state.update(status='complete',finished_utc=common.utc())
        save()
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        save()
        raise
    finally:
        guard.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    run(parser.parse_args().data_dir)
