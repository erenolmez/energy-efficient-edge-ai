"""Separate 200-epoch ImageNet initialization comparison; never opens test data."""
from contextlib import ExitStack
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from hpc.clean_protocol import atomic_write, file_hash, fingerprint, validate_resume, validate_split, write_json
from hpc.full_factorial_protocol import checked_plan, trial_directory
from hpc.train_classifier_pilot import run_lock, seed_worker
from hpc.train_full_factorial import cutmix, make_optimizer, make_scheduler, state_fingerprint
from hpc.wandb_tracking import public_config, tracking_session

CASES = (312,329,314,139,228,237,1678,415)
SOURCE_PLAN = '9b3173c990036c44f2a1b7b09d1d38000a82cf2c03f796ff4d58047b8153f50c'
STUDY = 'imagenet32_128_matched_recipes_v1'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','train'))
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--case',type=int,choices=CASES)
    p.add_argument('--size',type=int,choices=(32,128))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run on a Slurm allocation, not a login node')
    if args.action=='train' and (args.case is None or args.size is None):
        p.error('Training requires --case and --size')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    import torchvision
    import timm
    from model_pipeline.models import create_classifier
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR100

    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if args.action=='train':
            torch.cuda.manual_seed_all(seed)

    scratch=args.base/'scratch32_full_factorial_v1'
    frozen=checked_plan(scratch/'plan.json')
    if frozen['sha256'] != SOURCE_PLAN or frozen['epochs'] != 200:
        raise ValueError('Frozen scratch plan mismatch')
    split=load(args.base/'split_seed42.json')
    raw=CIFAR100(args.data_dir,train=True,download=False)
    validate_split(split,raw.targets)
    ready=load(scratch/'data_ready.json')
    import hashlib
    if (ready['plan_sha256'] != SOURCE_PLAN or ready['split_sha256'] != split['sha256']
        or ready['training_pixels_sha256'] != hashlib.sha256(raw.data.tobytes()).hexdigest()):
        raise ValueError('Dataset or split does not match the frozen scratch receipt')

    root=args.base/STUDY
    repo=Path(__file__).resolve().parents[1]
    source_names=('hpc/train_imagenet_comparison.py','hpc/train_full_factorial.py','hpc/clean_protocol.py',
                  'hpc/train_classifier_pilot.py','hpc/wandb_tracking.py','model_pipeline/models.py')
    hashes={name:file_hash(repo/name) for name in source_names}
    versions=dict(torch=str(torch.__version__),torchvision=torchvision.__version__,timm=timm.__version__,numpy=np.__version__)
    spec=dict(study=STUDY,scratch_plan_sha256=SOURCE_PLAN,split_sha256=split['sha256'],
              source_hashes=hashes,versions=versions,cases=list(CASES),sizes=[32,128],epochs=200,
              model='PC model registry, ImageNet initialization; same initial weights for both sizes',
              transform='native-to-target bilinear; pad size/8, random crop, flip; frozen CIFAR normalization',
              test_evaluated=False)
    receipt=root/'prepared.json'
    if args.action=='prepare':
        root.mkdir(parents=True,exist_ok=False)
        for cid in CASES:
            trial=frozen['trials'][cid]
            summary=load(trial_directory(scratch,cid,trial['seed'])/'summary.json')
            if (summary['status']!='complete' or summary['epochs_complete']!=200
                or summary['trial']!=trial or summary['plan_sha256']!=SOURCE_PLAN):
                raise ValueError(f'Scratch source case {cid} is not complete and compatible')
        weights={}
        for cid in CASES:
            trial=frozen['trials'][cid]
            seed_all(trial['seed'])
            model=create_classifier(trial['architecture'],pretrained=True)
            path=root/f'initial_{cid:04d}.pt'
            atomic_write(path,lambda dest:torch.save(model.state_dict(),dest))
            weights[str(cid)]=dict(file_sha256=file_hash(path),state_sha256=state_fingerprint(model.state_dict()),
                                  source_best_validation=load(trial_directory(scratch,cid,trial['seed'])/'summary.json')['best_validation_accuracy'])
            del model
            print(json.dumps(dict(event='initialized',case=cid,architecture=trial['architecture'])),flush=True)
        write_json(receipt,dict(spec=spec,weights=weights))
        return

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation required')
    prepared=load(receipt)
    if prepared['spec'] != spec:
        raise ValueError('Prepared versions, source, or split changed')
    trial=frozen['trials'][args.case]
    seed=trial['seed']; size=args.size
    initial_path=root/f'initial_{args.case:04d}.pt'
    initial=prepared['weights'][str(args.case)]
    if file_hash(initial_path)!=initial['file_sha256']:
        raise ValueError('Initial weight file hash mismatch')
    plan=dict(frozen,study=STUDY,image_size=size,architecture=trial['architecture'])
    plan['sha256']=fingerprint(prepared)
    recipe=dict(trial=trial,plan_sha256=plan['sha256'],image_size=size,initial=initial,versions=versions)
    rng_seed=lambda ep: seed_all(seed+ep)
    rng_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    model=create_classifier(trial['architecture'],pretrained=False)
    model.load_state_dict(torch.load(initial_path,map_location='cpu',weights_only=True))
    if state_fingerprint(model.state_dict())!=initial['state_sha256']:
        raise ValueError('Initial state changed')
    model=model.cuda()
    optimizer=make_optimizer(model,trial,plan)
    scheduler=make_scheduler(optimizer,trial,plan)
    scaler=torch.amp.GradScaler('cuda')
    criterion=torch.nn.CrossEntropyLoss()
    normalize=T.Normalize(plan['normalization_mean'],plan['normalization_std'])
    resize=[] if size==32 else [T.Resize((size,size),antialias=True)]
    transforms={'train':T.Compose([*resize,T.RandomCrop(size,padding=size//8),T.RandomHorizontalFlip(),T.ToTensor(),normalize]),
                'validation':T.Compose([*resize,T.ToTensor(),normalize])}
    subsets={role:Subset(CIFAR100(args.data_dir,train=True,download=False,transform=transform),
                split['indices']['classifier_'+role]) for role,transform in transforms.items()}
    directory=root/f'case_{args.case:04d}_{size}px'
    directory.mkdir(exist_ok=args.resume)
    latest=directory/'latest.pt'
    history=[]; best_state=None; best_acc=-1.; best_loss=float('inf'); stop=[]
    with run_lock(directory), ExitStack() as stack:
        if args.resume:
            checkpoint=torch.load(latest,map_location='cpu',weights_only=True)
            validate_resume(checkpoint,recipe,split['sha256'],hashes)
            if checkpoint['epoch'] >= 200:
                raise ValueError('Case has already reached epoch 200')
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
            history,best_state,best_acc,best_loss=(checkpoint[k] for k in ('history','best_state','best_accuracy','best_loss'))
            del checkpoint
        config=public_config(plan,trial,seed,f'imagenet_{size}px')
        config['initialization']='imagenet_pc_registry'
        tracker=stack.enter_context(tracking_session(directory,config,
            mode=os.environ.get('SCRATCH_TRACKING_MODE','disabled'),project=os.environ.get('WANDB_PROJECT'),
            entity=os.environ.get('WANDB_ENTITY')))
        def stop_after_epoch(signum,_frame):
            stop.append(signum)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
            signal.signal(sig,stop_after_epoch)
        write_json(directory/f'run_{time.time_ns()}.json',dict(recipe=recipe,job=os.environ['SLURM_JOB_ID']))
        for epoch in range(len(history)+1,201):
            rng_seed(epoch); torch.cuda.reset_peak_memory_stats()
            row=dict(epoch=epoch,lr=optimizer.param_groups[0]['lr'])
            for role,subset in subsets.items():
                training=role=='train'; model.train(training)
                loader=DataLoader(subset,batch_size=plan['batch_size'],shuffle=training,num_workers=plan['workers'],
                    pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed+epoch))
                torch.cuda.synchronize(); started=time.perf_counter()
                total=0; loss_sum=torch.zeros((),device='cuda'); correct=torch.zeros((),device='cuda')
                with torch.set_grad_enabled(training):
                    for x,y in loader:
                        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                        a,b,lam=y,y,1.
                        if training:
                            x,a,b,lam=cutmix(x,y,trial['cutmix_alpha'],plan['cutmix_probability'])
                            optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16):
                            pred=model(x); loss=lam*criterion(pred,a)+(1-lam)*criterion(pred,b)
                        if training:
                            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                        loss_sum+=loss.detach()*len(y); total+=len(y)
                        correct+=lam*(pred.argmax(1)==a).sum()+(1-lam)*(pred.argmax(1)==b).sum()
                torch.cuda.synchronize()
                row[role]=dict(images=total,loss=loss_sum.item()/total,accuracy=100*correct.item()/total,
                               seconds=time.perf_counter()-started)
                if not math.isfinite(row[role]['loss']):
                    raise RuntimeError('Non-finite loss; previous checkpoint remains intact')
            row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            v=row['validation']
            if (v['accuracy'],-v['loss'])>(best_acc,-best_loss):
                best_acc,best_loss=v['accuracy'],v['loss']
                best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
            scheduler.step(); history.append(row)
            atomic_write(latest,lambda dest:torch.save(dict(epoch=epoch,recipe=recipe,split_sha256=split['sha256'],
                source_hashes=hashes,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),history=history,best_state=best_state,best_accuracy=best_acc,best_loss=best_loss),dest))
            tracker.log(row)
            print(json.dumps(dict(event='epoch_saved',case=args.case,size=size,**row)),flush=True)
            if stop:
                break
        result=dict(status='complete' if len(history)==200 else 'paused',epochs_complete=len(history),
                    best_validation_accuracy=best_acc,best_validation_loss=best_loss,test_evaluated=False,
                    training_seconds=sum(r['train']['seconds']+r['validation']['seconds'] for r in history),
                    recipe=recipe,checkpoint_sha256=file_hash(latest))
        write_json(directory/'summary.json',result)
        tracker.summary(result)


if __name__=='__main__':
    main()
