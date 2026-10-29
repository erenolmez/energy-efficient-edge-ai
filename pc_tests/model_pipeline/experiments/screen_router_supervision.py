"""Seed42 validation-only router data/calibration/epochs/loss screen; no testdata."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import gc,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.ablation_design import nested_subsets,positive_weights
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness

REPORT=PIPELINE/'reports/router_supervision_screen_20261029/study.json'
ARTIFACTS=PIPELINE/'artifacts/router_supervision_screen_20261029'

def main():
    if REPORT.exists():raise RuntimeError('Saved supervision screen exists; no blind restart')
    wait(__file__)
    import torch
    from torch.nn import functional as F
    from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    predecessor=PIPELINE/'reports/compact_switching_confirmation_20261028/audited_summary.json'
    assert json.loads(predecessor.read_text())['audit_passed']
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,targets,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    names=[r['name'] for r in records];fit,cal=split_validation(labels,42)
    subsets=nested_subsets(labels,fit,[600,1000,2000,3000],42);calsets=nested_subsets(labels,cal,[500,1000,2000],43)
    tune=subsets[600];internal=np.array([p for p in fit if p not in set(tune.tolist())]);assert len(internal)==2400
    original=json.loads((PIPELINE/'artifacts/pc_routers_correctness/global_correctness_seed42.json').read_text())
    assert original['fit_positions']==fit.tolist() and original['calibration_positions']==cal.tolist()
    profiles=json.loads((PIPELINE/'reports/compact_runtime_profile_20261028/study.json').read_text());pool=profiles['pool']
    summary=json.loads((PIPELINE/'reports/compact_runtime_profile_20261028/audited_summary.json').read_text())
    columns=[names.index(n) for n in pool];base=pool.index(baseline)
    costs=np.array([next(r['eager_ms'] for r in summary['models'] if r['name']==n) for n in pool])
    targets=targets.copy()
    for col,name in zip(columns,pool):
        fresh=profiles['fresh_original_validation'][name];assert digest(fresh['predictions_path'])==fresh['predictions_sha256']
        targets[:,col]=np.load(fresh['predictions_path'],allow_pickle=False)==labels
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    jobs=[dict(name=f'fit{n}_epochs20',positions=subsets[n],epochs=20,balanced=False,internal=False) for n in (1000,2000,3000)]
    jobs += [dict(name=f'internal_{kind}_epochs60',positions=internal,epochs=60,balanced=kind=='balanced',internal=True) for kind in ('unweighted','balanced')]
    report=dict(status='running',results=[],fits=[],test_data_opened=False,seed=42,source_sha256=digest(Path(__file__)),
        predecessor_sha256=digest(predecessor),provenance=provenance,models=names,policy_pool=pool,fit_positions=fit.tolist(),calibration_positions=cal.tolist(),
        internal_tune_positions=tune.tolist(),expected_fits=5,expected_epoch_checkpoints=180,
        protocol='Predefined seed42 validation-only screening. Nestedclass-balanced1000/2000/3000fits20epochs; separate2400internaltrain/600internaltune models compare20/60epochs unweighted vsbalancedBCE. Candidate213targets cachedexcept6currentoriginal-enginecolumns refreshed. No newdata/classifierretraining. Select20or60 checkpoint onlybyinternal600poolBrier before policycalibration. Policycal500/1000/2000reservedimages, remainder heldoutvalidation whereavailable; no testdata ordeploymentclaims.',
        balanced_definition='pos_weight=clip((n-positive)/max(positive,1),0.1,10), fit-only perhead; inference logits subtractlog(pos_weight) as predefined odds correction, not fitted oncal/test.',
        protected_hashes={str(p):digest(p) for p in [PIPELINE/'reports/pc_routes_correctness.json',PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/pc_routers_correctness/global_correctness_seed42.pth']},
        created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='supervision_screen',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report)
    buffers=SharedBatchInputs(maximum_batch=64);y=torch.from_numpy(targets.astype(np.float32));meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVML energy unavailable')
    try:
        for job in jobs:
            check(__file__);torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=np.random.default_rng(42)
            positions=job['positions'];weight=positive_weights(targets[positions]) if job['balanced'] else np.ones(213,dtype=np.float32)
            tensorweight=torch.from_numpy(weight).cuda();meter.start();start=time.perf_counter()
            model=CorrectnessRouter(213).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[];snapshots={};tune_metrics={}
            def evaluate(pos):
                model.eval();values=[]
                with torch.inference_mode():
                    for i in range(0,len(pos),64):
                        batch=pos[i:i+64];h=torch.from_numpy(np.array(images[batch])).pin_memory()
                        logits=model(buffers.prepare(h));values.append(torch.sigmoid(logits-tensorweight.log()).cpu().numpy())
                return np.concatenate(values)
            for epoch in range(1,job['epochs']+1):
                check(__file__);model.train();losses=[];permutation=rng.permutation(positions)
                for first in range(0,len(permutation),64):
                    batch=permutation[first:first+64];h=torch.from_numpy(np.array(images[batch])).pin_memory()
                    logits=model(buffers.prepare(h));loss=F.binary_cross_entropy_with_logits(logits,y[batch].cuda(),pos_weight=tensorweight)
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
                checkpoint=ARTIFACTS/f'{job["name"]}_latest.pth';temporary=checkpoint.with_suffix('.pth.tmp')
                torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,fit_positions=positions.tolist(),weight=weight),temporary);temporary.replace(checkpoint)
                if epoch in (20,60):
                    snapshot=ARTIFACTS/f'{job["name"]}_epoch{epoch}.pth';torch.save(model.state_dict(),snapshot);snapshots[epoch]=str(snapshot)
                    if job['internal']:
                        q=evaluate(tune);tune_metrics[epoch]=dict(pool_brier=float(np.mean((q[:,columns]-targets[tune][:,columns])**2)),all213_brier=float(np.mean((q-targets[tune])**2)))
                done(f'{job["name"]}_epoch{epoch}',kind='epoch',fit=job['name'],epoch=epoch,loss=history[-1],checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
            torch.cuda.synchronize();energy=meter.finish(len(positions)*job['epochs'])
            if energy.get('gpu_energy_j') is None:raise RuntimeError('FitGPUenergy unavailable')
            chosen=min(tune_metrics,key=lambda e:(tune_metrics[e]['pool_brier'],e)) if job['internal'] else 20
            model.load_state_dict(torch.load(snapshots[chosen],map_location='cuda',weights_only=True));q=evaluate(cal)
            policyrows=[]
            for n,subset in calsets.items():
                mask=np.isin(cal,subset);holdout=~mask
                policy=calibrate_correctness(q[mask][:,columns],targets[cal[mask]][:,columns].astype(float),np.ones(n),costs,base,1.)
                validation_accuracy=None;reference_accuracy=None
                if holdout.any():
                    choices=np.full(holdout.sum(),base,dtype=np.int64) if policy['fallback'] else select_correctness(q[holdout][:,columns],costs,policy['penalty'])
                    yy=targets[cal[holdout]][:,columns];validation_accuracy=100*float(yy[np.arange(len(choices)),choices].mean());reference_accuracy=100*float(yy[:,base].mean())
                policyrows.append(dict(calibration_images=n,calibration_positions=subset.tolist(),policy=policy,heldout_validation_images=int(holdout.sum()),
                    heldout_validation_accuracy=validation_accuracy,heldout_reference_accuracy=reference_accuracy,
                    heldout_validation_accuracy_pass=None if validation_accuracy is None else reference_accuracy-validation_accuracy<=1.+1e-10))
            out=ARTIFACTS/f'{job["name"]}_calibration.npz';np.savez_compressed(out,positions=cal,probabilities=q,correctness=targets[cal])
            row=dict(name=job['name'],fit_positions=positions.tolist(),fit_images=len(positions),epochs=job['epochs'],balanced=job['balanced'],losses=history,
                weights=snapshots[chosen],weights_sha256=digest(snapshots[chosen]),selected_epoch=chosen,snapshots={str(k):dict(path=v,sha256=digest(v)) for k,v in snapshots.items()},
                internal_tune_metrics=tune_metrics,gpu_fit_energy=energy,training_scope='GPUonly; includesinit,fit,epochcheckpointwrites andinternalvalidation whenused; excludesfinalpolicycal/export/test/CPUenergy',
                calibration_file=str(out),calibration_sha256=digest(out),pool_brier=float(np.mean((q[:,columns]-targets[cal][:,columns])**2)),all213_brier=float(np.mean((q-targets[cal])**2)),policies=policyrows)
            report['fits'].append(row);done('fit_'+job['name'],kind='fit',name=job['name']);print('FIT '+job['name']+' '+str(row['pool_brier']),flush=True)
            del model,optimizer,tensorweight;gc.collect();torch.cuda.empty_cache()
        assert len(report['fits'])==5
        for p,h in report['protected_hashes'].items():assert digest(p)==h
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
