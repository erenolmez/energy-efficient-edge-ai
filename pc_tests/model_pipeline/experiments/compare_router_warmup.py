"""Isolated36-fit warm-up study; no classifiers, policy tuning or test access."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.experiments.router_warmup_design import slots,ready,require_energy

ART=PIPELINE/'artifacts/router_warmup_20261103'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def child(index):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    from pc_tests.model_pipeline.experiments import diagnose_batch_plateau as logger
    plan=json.loads((ART/'plan.json').read_text())
    slot=plan['slots'][index]
    directory=ART/slot['id']
    directory.mkdir(exist_ok=False)
    logger.OUT=directory # Reuse frozen telemetry code, with a NEW artifact directory.
    telemetry=logger.Telemetry()
    nv,h=telemetry.nv,telemetry.handle
    temperature=nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU)
    reasons=nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
    if not ready(temperature,reasons):
        atomic_json(directory/'deferred.json',dict(status='deferred_no_retry',temperature_c=temperature,reasons=reasons))
        raise RuntimeError('Predefined temperature/thermal-flag start gate failed')
    telemetry.thread.start()
    result=dict(**slot,status='running',pid=os.getpid(),executable=sys.executable,started_utc=utc(),start_temperature_c=temperature,
                start_clock_reasons=reasons,epochs=[],test_data_accessed=False)
    atomic_json(directory/'result.json',result)
    try:
        for path, expected in plan['protected_sha256'].items():
            if digest(path)!=expected:
                raise RuntimeError('Protected source/data/weights changed: '+path)
        cache=PIPELINE/'artifacts/pc_precision/data_cache'
        labels=np.load(cache/'validation_labels.npy',allow_pickle=False)
        images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
        correct=np.load(ART/'correctness_targets.npy',allow_pickle=False)
        prior=json.loads((PIPELINE/f'artifacts/pc_routers_correctness/global_correctness_seed{slot["seed"]}.json').read_text())
        fit=np.asarray(prior['fit_positions'])
        from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
        wanted,cal=split_validation(labels,slot['seed'])
        if not np.array_equal(fit,wanted) or len(fit)!=3000 or len(cal)!=2000 or prior['settings']['epochs']!=20:
            raise RuntimeError('Original fit split/recipe mismatch')
        torch.set_num_threads(1)
        torch.cuda.init()
        buffers=SharedBatchInputs()
        torch.manual_seed(slot['seed']);torch.cuda.manual_seed_all(slot['seed'])
        random.seed(slot['seed']);np.random.seed(slot['seed'])
        result.update(gpu=nv.nvmlDeviceGetName(h),driver=nv.nvmlSystemGetDriverVersion(),torch_version=torch.__version__)
        # Dummy work has separate energy/time and cannot modify actual RNG/order.
        result['warmup']=dict(status='not_applicable_cold_control')
        if slot['condition']=='warm':
            saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            telemetry.window='dummy_warmup'
            dummy=CorrectnessRouter(213).cuda()
            opt=torch.optim.Adam(dummy.parameters(),lr=.001)
            host=torch.zeros((64,3,128,128),dtype=torch.uint8).pin_memory()
            target=torch.zeros((64,213),device='cuda')
            meter=EnergyCounter();meter.start();started=time.monotonic()
            for step in range(100):
                output=dummy(buffers.prepare(host))
                loss=F.binary_cross_entropy_with_logits(output,target)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            torch.cuda.synchronize()
            result['warmup']=dict(steps=100,seconds=time.monotonic()-started,energy=meter.finish(6400))
            del dummy,opt,host,target,output,loss
            random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])
            result['warmup_rng_restored']=torch.equal(torch.get_rng_state(),saved[2]) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved[3]))
            if not result['warmup_rng_restored']:
                raise RuntimeError('Warm-up changed actual-fit RNG')
        telemetry.check()
        telemetry.window='fit_initialization'
        meter=EnergyCounter()
        if not meter.available:
            raise RuntimeError('GPU fit energy unavailable')
        torch.cuda.synchronize();meter.start();fit_start=time.monotonic()
        model=CorrectnessRouter(213)
        result['initial_weights_sha256']=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        model=model.cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        targets=torch.from_numpy(correct.astype(np.float32))
        rng=np.random.default_rng(slot['seed'])
        for epoch in range(20):
            telemetry.check()
            telemetry.window=f'epoch_{epoch+1}'
            begin=time.monotonic();model.train();order=rng.permutation(fit)
            if epoch==0:
                result['first_order_sha256']=hashlib.sha256(order.tobytes()).hexdigest()
            losses=[]
            for start in range(0,len(order),64):
                positions=order[start:start+64]
                normalized=buffers.prepare(torch.from_numpy(np.array(images[positions])).pin_memory())
                logits=model(normalized)
                loss=F.binary_cross_entropy_with_logits(logits,targets[positions].cuda())
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                losses.append(float(loss.detach()))
            torch.cuda.synchronize();training_end=time.monotonic()
            checkpoint=directory/f'epoch_{epoch+1:02d}.pth'
            temporary=checkpoint.with_suffix('.pth.tmp')
            torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                            settings=prior['settings'],provenance=plan['provenance'],rng_state=rng.bit_generator.state),temporary)
            temporary.replace(checkpoint)
            atomic_json(checkpoint.with_suffix('.json'),dict(status='complete',epoch=epoch+1,seed=slot['seed'],checkpoint=str(checkpoint),sha256=digest(checkpoint)))
            result['epochs'].append(dict(epoch=epoch+1,bce=float(np.mean(losses)),training_seconds=training_end-begin,
                                          checkpoint_seconds=time.monotonic()-training_end))
            # Epoch results persisted inside the fit boundary, equally in both conditions.
            atomic_json(directory/'result.json',result)
            telemetry.check()
        torch.cuda.synchronize()
        result['fit_energy']=require_energy(meter.finish(20*len(fit)))
        result['fit_window_seconds']=time.monotonic()-fit_start
        result['final_checkpoint_sha256']=digest(checkpoint)
        result['status']='complete'
        result['finished_utc']=utc()
    except Exception as exc:
        result.update(status='stopped_preserve_no_retry',error=repr(exc))
        raise
    finally:
        telemetry.done.set();telemetry.thread.join(timeout=3)
        atomic_json(directory/'result.json',result)


def controller():
    import numpy as np
    from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation
    if (ART/'run.lock').exists():
        raise RuntimeError('Existing controller reservation; no automatic resume')
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'run.lock').open('x') as f:
        f.write(json.dumps(dict(pid=os.getpid(),executable=sys.executable,utc=utc())))
    state=dict(status='preparing',results=[],test_data_accessed=False)
    atomic_json(ART/'study.json',state)
    try:
        records,labels,correct,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',PIPELINE/'artifacts/pc_precision/data_cache')
        np.save(ART/'correctness_targets.npy',correct,allow_pickle=False)
        protected=list((PIPELINE/'artifacts/pc_routers_correctness').glob('global_correctness_seed*.*'))
        protected += [PIPELINE/'reports/gpu_static.json',PIPELINE/'reports/pc_routes_correctness.json',
                      PIPELINE/'artifacts/correctness_training_energy_20261028/training_energy.json',
                      PIPELINE/'artifacts/pc_precision/data_cache/validation_rgb128.npy',PIPELINE/'artifacts/pc_precision/data_cache/validation_labels.npy',
                      ART/'correctness_targets.npy',Path(__file__),Path(__file__).with_name('router_warmup_design.py'),
                      PIPELINE/'experiments/diagnose_batch_plateau.py',PIPELINE/'routers/correctness/correctness_router.py',
                      PIPELINE/'runtime/gpu_batch_inputs.py',PIPELINE/'runtime/gpu_energy.py',PIPELINE/'runtime/atomic_checkpoint.py']
        plan=dict(slots=slots(),epochs=20,batch_size=64,optimizer='Adam',lr=.001,warmup_steps=100,idle_seconds=60,
                  provenance=provenance,protected_sha256={str(p):digest(p) for p in protected},
                  scope='GPU fit energy only, excludes CPU/restPC; warm-up separate; no test access',
                  boundary='Initialization,20 epochs and checkpoint/result writes inside fit window; CUDA context and validation preparation outside. Initial hash/result instrumentation adds equal overhead versus historical code; not byte-identical timing recipe.')
        atomic_json(ART/'plan.json',plan)
        state['plan_sha256']=digest(ART/'plan.json')
        state['status']='running'
        for index,slot in enumerate(plan['slots']):
            if index:
                state.update(phase='fixed_idle_60s',idle_started_utc=utc())
                atomic_json(ART/'study.json',state)
                time.sleep(60) # Background study interval, not interactive polling.
            state.update(phase='fresh_child',current=slot)
            atomic_json(ART/'study.json',state)
            with (ART/(slot['id']+'.stdout.log')).open('x') as stdout,(ART/(slot['id']+'.stderr.log')).open('x') as stderr:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--slot',str(index)],stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                # Capture exact Windows executable/script/creation identity, not PID alone.
                command=f'Get-CimInstance Win32_Process -Filter "ProcessId = {process.pid}" | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress'
                identity=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
                atomic_json(ART/(slot['id']+'.identity.json'),json.loads(identity.stdout))
                state['active_child_pid']=process.pid
                atomic_json(ART/'study.json',state)
                code=process.wait()
            if code!=0:
                raise RuntimeError('Child failed/deferred: '+slot['id']+'; preserve logs, no retry')
            result=json.loads((ART/slot['id']/'result.json').read_text())
            if result['status']!='complete' or len(result['epochs'])!=20:
                raise RuntimeError('Incomplete child result')
            state['results'].append(result)
            state['active_child_pid']=None
            atomic_json(ART/'study.json',state)
        if any(digest(p)!=h for p,h in plan['protected_sha256'].items()):
            raise RuntimeError('Protected files changed')
        state['status']='complete_pending_audit'
    except Exception as exc:
        state.update(status='stopped_preserve_no_retry',error=repr(exc))
        raise
    finally:
        atomic_json(ART/'study.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int);args=p.parse_args()
    controller() if args.slot is None else child(args.slot)
