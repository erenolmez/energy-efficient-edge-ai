"""Matched five-epoch Adam recovery curves on one freshly pruned PC checkpoint."""
import argparse
import copy
import hashlib
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
from pc_tests.model_pipeline.experiments.prune_finetune import make_loader, structurally_prune
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE/'reports/adam_recovery_curve_20261105'
MODELS = PIPELINE/'artifacts/adam_recovery_curve_20261105'
RATES = (.01,.001)
EPOCHS = 5


def parameter_hash(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate(model, loader, check):
    model.eval()
    loss = 0.
    counts = torch.zeros((100,100),device='cuda',dtype=torch.int64)
    with torch.inference_mode():
        for x,y in loader:
            x,y = x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            z = model(x)
            assert z.shape==(len(y),100) and torch.isfinite(z).all()
            loss += float(torch.nn.functional.cross_entropy(z,y,reduction='sum'))
            counts += torch.bincount(100*y+z.argmax(1),minlength=10000).reshape(100,100)
            check()
    a = counts.cpu().numpy()
    assert np.array_equal(a.sum(1),np.full(100,50))
    return dict(accuracy=100*int(a.trace())/5000,loss=loss/5000,confusion=a.tolist())


def run(data_dir):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f:
        json.dump(dict(pid=os.getpid(),utc=common.utc()),f)
    common.OUT=OUT
    guard=common.Guard()
    state=dict(status='preparing',architecture='resnet18',seed=42,image_size=128,
               pruning_ratio=.30,rates=RATES,epochs=EPOCHS,optimizer='Adam',
               train_images=45000,validation_images=5000,batch=64,amp=True,
               label_smoothing=.1,weight_decay=0,scheduler=None,test_data_accessed=False,rows=[])

    def save():
        atomic_json(OUT/'study.json',state)

    def check():
        guard.check()
        if torch.cuda.mem_get_info()[0]<1024**3:
            raise RuntimeError('Below 1 GiB VRAM reserve')

    try:
        state['start']=guard.start()
        torch.set_num_threads(1)
        seed_everything(42)
        baseline=PIPELINE/'artifacts/baselines/resnet18__p0__fp32.pth'
        split_path=baseline.parent/'cifar100_split_seed42.json'
        pilot_path=PIPELINE/'reports/adam_post_prune_pilot_v2_20261105/study.json'
        split,pilot=(json.loads(p.read_text()) for p in (split_path,pilot_path))
        assert pilot['status']=='complete' and pilot['split_sha256']==split['sha256']
        protected={str(p):common.digest(p) for p in (baseline,split_path,pilot_path,
            Path(__file__),PIPELINE/'experiments/prune_finetune.py',
            PIPELINE/'experiments/adam_post_prune_pilot.py')}
        assert protected[str(baseline)]==pilot['protected_sha256'][str(baseline)]
        state['protected_sha256']=protected
        train_data=TrainOnlyCIFAR100(root=data_dir,train=True,download=False,transform=training_transform(128))
        validation_data=copy.copy(train_data)
        validation_data.transform=evaluation_transform(128)
        train,val=stratified_indices(train_data.targets,5000,42)
        assert train==split['train_indices'] and val==split['validation_indices']
        assert split_fingerprint(train,val)==split['sha256']
        state.update(split_sha256=split['sha256'],class_names=train_data.classes,gpu=torch.cuda.get_device_name(0))
        val_loader=make_loader(Subset(validation_data,val),64,2,False,43)
        model=torch.load(baseline,map_location='cpu',weights_only=False).cuda().eval()
        guard.phase='fresh_pruning'
        args=SimpleNamespace(image_size=128,round_to=8,global_pruning=True,isomorphic=True,max_pruning_ratio=.95)
        state['complexity']=structurally_prune(model,.30,args,torch.device('cuda'))
        assert state['complexity']==pilot['complexity']
        state['pruned']=validate(model,val_loader,check)
        assert state['pruned']['confusion']==pilot['pruned']['confusion']
        model=model.cpu()
        state['initial_state_sha256']=parameter_hash(model)
        state['historical_20_epoch_sgd_validation_accuracy']=pilot['historical_20_epoch_sgd_validation_accuracy']
        MODELS.mkdir(parents=True,exist_ok=True)
        state['status']='running'
        save()
        for lr in RATES:
            seed_everything(42)
            loader=make_loader(Subset(train_data,train),64,2,True,42)
            candidate=copy.deepcopy(model).cuda()
            assert parameter_hash(candidate)==state['initial_state_sha256']
            optimizer=torch.optim.Adam(candidate.parameters(),lr=lr)
            scaler=torch.amp.GradScaler('cuda')
            loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=.1)
            for epoch in range(1,EPOCHS+1):
                guard.phase=f'lr_{lr}_epoch_{epoch}_train'
                candidate.train()
                total=correct=skipped=0
                losses=0.
                for step,(x,y) in enumerate(loader,1):
                    x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        z=candidate(x)
                        loss=loss_fn(z,y)
                    value=float(loss.detach())
                    if not math.isfinite(value):
                        raise RuntimeError('Nonfinite training loss')
                    scale=scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped+=scaler.get_scale()<scale
                    total+=len(y); correct+=int((z.argmax(1)==y).sum()); losses+=value*len(y)
                    check()
                assert total==45000 and step==704
                guard.phase=f'lr_{lr}_epoch_{epoch}_validation'
                result=validate(candidate,val_loader,check)
                path=MODELS/f'adam_lr{lr}_epoch{epoch}.pth'
                with path.open('xb') as f:
                    torch.save(copy.deepcopy(candidate).cpu().eval(),f)
                state['rows'].append(dict(lr=lr,epoch=epoch,training_images=total,steps=step,
                    skipped_amp_updates=skipped,train_accuracy=100*correct/total,train_loss=losses/total,
                    validation=result,checkpoint=str(path),checkpoint_sha256=common.digest(path)))
                save()
                print(f"Adam LR {lr:g} epoch {epoch}/{EPOCHS}: {result['accuracy']:.2f}% validation",flush=True)
            del loader,optimizer,scaler,candidate
            torch.cuda.empty_cache()
        assert all(common.digest(Path(p))==h for p,h in protected.items())
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
