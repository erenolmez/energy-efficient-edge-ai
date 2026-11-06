"""Validation-only one-epoch Adam recovery pilot, using historical PC weights."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from torchvision.datasets import CIFAR100
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as common

OUT = PIPELINE/'reports/adam_post_prune_pilot_v2_20261105'
ARTIFACTS = PIPELINE/'artifacts/adam_post_prune_pilot_v2_20261105'


class TrainOnlyCIFAR100(CIFAR100):
    # torchvision integrity checking normally checks both archives. The class
    # is module-level so Windows data loader workers can pickle the dataset.
    test_list = []


def main(data_dir):
    import copy
    import torch
    import torch_pruning
    import torchvision
    from torch.utils.data import Subset
    from model_pipeline.data import (evaluation_transform, training_transform,
                                    split_fingerprint, stratified_indices)
    from model_pipeline.reproducibility import seed_everything
    from pc_tests.model_pipeline.experiments.prune_finetune import make_loader, structurally_prune

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('x') as lock:
        json.dump(dict(pid=os.getpid(), utc=common.utc()), lock)
    common.OUT = OUT
    guard = common.Guard()
    state = dict(status='preparing', architecture='resnet18', requested_prune_percent=30,
                 image_size=128, seed=42, epochs=1, optimizer='Adam', learning_rate=.01,
                 batch_size=64, amp=True, weight_decay=0, label_smoothing=.1,
                 dataset='CIFAR-100', test_data_accessed=False,
                 prior_attempt='adam_post_prune_pilot_20261105 stopped before validation due to worker pickling; no training',
                 torch=torch.__version__, torchvision=torchvision.__version__,
                 torch_pruning=str(getattr(torch_pruning,'__version__','unknown')))

    def save():
        atomic_json(OUT/'study.json', state)

    def check():
        guard.check()
        if torch.cuda.mem_get_info()[0] < 1024**3:
            raise RuntimeError('Below 1 GiB free VRAM; stopped without retry')

    def evaluate(model, loader):
        model.eval()
        matrix = torch.zeros((100,100),device='cuda',dtype=torch.int64)
        total_loss = 0.
        with torch.inference_mode():
            for image,target in loader:
                guard.check()
                image,target = image.cuda(non_blocking=True),target.cuda(non_blocking=True)
                logits = model(image)
                if logits.shape != (len(target),100) or not torch.isfinite(logits).all():
                    raise RuntimeError('Invalid validation logits')
                total_loss += float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
                matrix += torch.bincount(100*target+logits.argmax(1),minlength=10000).reshape(100,100)
                check()
        counts = matrix.cpu().numpy()
        totals = counts.sum(1)
        assert np.array_equal(totals,np.full(100,50))
        correct = np.diag(counts)
        return dict(accuracy=100.*int(correct.sum())/int(totals.sum()),
                    loss=total_loss/int(totals.sum()), confusion=counts.tolist(),
                    correct_per_class=correct.tolist(), per_class_accuracy=(100*correct/totals).tolist())

    try:
        state['initial_telemetry'] = guard.start()
        torch.set_num_threads(1)
        seed_everything(42)
        args = SimpleNamespace(image_size=128, round_to=8, global_pruning=True,
                               isomorphic=True, max_pruning_ratio=.95)
        root = PIPELINE/'artifacts/baselines'
        baseline = root/'resnet18__p0__fp32.pth'
        meta = baseline.with_suffix('.json')
        split_path = root/'cifar100_split_seed42.json'
        historical = PIPELINE/'artifacts/pruned/resnet18__p30__fp32.json'
        historical_history = PIPELINE/'artifacts/pruned/resnet18__p30__fp32_history.csv'
        state['protected_sha256'] = {str(p):common.digest(p) for p in
            (baseline,meta,split_path,historical,historical_history,Path(__file__),
             PIPELINE/'experiments/prune_finetune.py')}
        source, split = (json.loads(p.read_text()) for p in (meta,split_path))
        assert source['architecture']=='resnet18' and source['image_size']==128 and not source['smoke_test']
        train_data = TrainOnlyCIFAR100(root=data_dir,train=True,download=False,transform=training_transform(128))
        validation_data = copy.copy(train_data)
        validation_data.transform = evaluation_transform(128)
        train, val = stratified_indices(train_data.targets,5000,42)
        assert train==split['train_indices'] and val==split['validation_indices']
        assert split_fingerprint(train,val)==split['sha256']==source['split_sha256']
        assert len(train)==45000 and len(val)==5000
        state.update(split_sha256=split['sha256'], class_names=train_data.classes,
                     train_images=len(train), validation_images=len(val), gpu=torch.cuda.get_device_name(0))
        train_loader = make_loader(Subset(train_data,train),64,2,True,42)
        validation_loader = make_loader(Subset(validation_data,val),64,2,False,43)
        model = torch.load(baseline,map_location='cpu',weights_only=False).cuda().eval()
        guard.phase = 'unpruned_validation'
        state['unpruned'] = evaluate(model,validation_loader)
        if abs(state['unpruned']['accuracy']-source['best_validation_accuracy']) > .2:
            raise RuntimeError('Unpruned checkpoint did not reproduce the original validation result')
        guard.phase = 'prune'
        state['complexity'] = structurally_prune(model,.30,args,torch.device('cuda'))
        old = json.loads(historical.read_text())
        assert old['split_sha256']==split['sha256']
        assert all(state['complexity'][k]==old[k] for k in
                   ('baseline_parameters','pruned_parameters','baseline_macs','pruned_macs'))
        guard.phase = 'pruned_validation'
        state['pruned'] = evaluate(model,validation_loader)
        state['historical_20_epoch_sgd_validation_accuracy'] = old['best_validation_accuracy']
        print(f"Validation before recovery: {state['pruned']['accuracy']:.2f}%",flush=True)
        save()

        guard.phase = 'adam_epoch_1'
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        scaler = torch.amp.GradScaler('cuda')
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
        started = time.perf_counter()
        total = correct = 0
        losses = 0.
        for step,(image,target) in enumerate(train_loader,1):
            guard.check()
            image,target = image.cuda(non_blocking=True),target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(image)
                loss = criterion(logits,target)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += len(target)
            correct += int((logits.argmax(1)==target).sum())
            losses += value*len(target)
            check()
            if step%100==0:
                state['progress'] = dict(steps=step,images=total,mean_loss=losses/total)
                save()
                print(f'Adam epoch 1: {total}/45000 images',flush=True)
        assert total==45000 and step==len(train_loader)
        state['training'] = dict(images=total, steps=step, elapsed_s=time.perf_counter()-started,
                                 accuracy=100*correct/total, loss=losses/total)
        save()
        guard.phase = 'adam_validation'
        state['adam'] = evaluate(model,validation_loader)
        assert sum(p.numel() for p in model.parameters())==state['complexity']['pruned_parameters']
        for p,digest in state['protected_sha256'].items():
            assert common.digest(Path(p))==digest,p
        ARTIFACTS.mkdir(parents=True,exist_ok=True)
        destination=ARTIFACTS/'resnet18__p30__adam_epoch1.pth'
        with destination.open('xb') as handle:
            torch.save(model.cpu().eval(),handle)
        state['checkpoint'] = dict(path=str(destination),sha256=common.digest(destination))
        state.update(status='complete',finished_utc=common.utc())
        save()
        print(f"Validation after one Adam epoch: {state['adam']['accuracy']:.2f}%",flush=True)
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        save()
        raise
    finally:
        guard.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    main(parser.parse_args().data_dir)
