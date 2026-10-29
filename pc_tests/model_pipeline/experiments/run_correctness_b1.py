"""Isolated batch-one export, validation calibration, pilot and measured comparison."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gc
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation, digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json, costs_for
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches, split_validation

ARTIFACTS = PIPELINE / 'artifacts/correctness_b1_20261028'
REPORTS = PIPELINE / 'reports/correctness_b1_20261028'
STATE = PIPELINE / 'artifacts/pc_routers/thermal_guard.json'
SEEDS = (42,45,48)
BUDGETS = (.5,1.,1.5,2.)


def thermal():
    for attempt in range(20):
        try:
            state = json.loads(STATE.read_text(encoding='utf-8-sig'))
            break
        except (OSError, json.JSONDecodeError):
            if attempt == 19:
                raise RuntimeError('Persistently unreadable thermal state')
            time.sleep(.025)
    if state.get('status') != 'monitoring' or state.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal latch: preserve results, never restart')
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
    if not 0 <= age <= 130 or state.get('expected_script') != Path(__file__).name or os.getpid() not in state.get('tracked_process_ids',[]):
        raise RuntimeError('Fresh exact worker thermal guard required')


def expected_blocks():
    return len(SEEDS)*2*len(BUDGETS)*5*3


def validate_meter(row, images):
    for key in ('gpu_energy_j', 'gpu_energy_j_per_image', 'rgb_input_to_prediction_seconds'):
        if row.get(key) is None or not 0 < row[key] < float('inf'):
            raise RuntimeError('Missing/nonpositive finite measurement: '+key)
    if row['test_images'] != images or sum(row['selected_images']) != images:
        raise RuntimeError('Incomplete sweep or candidate counts')


def main():
    if (REPORTS / 'study.json').exists():
        raise RuntimeError('Existing study preserved; do not duplicate or blindly restart')
    deadline = time.monotonic()+120
    print('Waiting for exact guard pid='+str(os.getpid()), flush=True)
    while True:
        try:
            thermal()
            break
        except RuntimeError:
            state = json.loads(STATE.read_text(encoding='utf-8-sig'))
            if state.get('status') != 'monitoring' or time.monotonic() >= deadline:
                raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter, BatchCorrectnessRouter
    from pc_tests.model_pipeline.runtime.router_tensorrt import build_router, RouterRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
    from pc_tests.model_pipeline.runtime.measure_pc_routes import measure
    from pc_tests.model_pipeline.experiments.run_pc_routes import eligible_static
    torch.set_num_threads(1)
    cache_dir = PIPELINE / 'artifacts/pc_precision/data_cache'
    static_dir = PIPELINE / 'artifacts/gpu_static'
    records, validation_labels, correct, baseline, provenance = load_validation(PIPELINE/'reports/gpu_static.json',static_dir,cache_dir)
    names = [r['name'] for r in records]
    by_name = {r['name']:r for r in records}
    base_index = names.index(baseline)
    protected_paths = [PIPELINE/'reports/gpu_static.json', PIPELINE/'reports/pc_routes_correctness.json',
        PIPELINE/'reports/correctness_training_energy_20261028/audited_summary.json']
    protected_paths += list((PIPELINE/'artifacts/pc_routers_correctness').glob('global_correctness_seed*.*'))
    protected = {str(p):digest(p) for p in protected_paths}
    dependencies = [Path(__file__), PIPELINE/'runtime/measure_pc_routes.py',PIPELINE/'runtime/router_tensorrt.py',
        PIPELINE/'runtime/global_candidate_pool.py',PIPELINE/'runtime/gpu_engine_preflight.py',
        PIPELINE/'runtime/gpu_batch_inputs.py',PIPELINE/'runtime/gpu_energy.py',
        PIPELINE/'routers/correctness/correctness_router.py',PIPELINE/'routers/correctness/correctness_policy.py',
        PIPELINE/'routers/scalar/batch_router_policy.py',PIPELINE/'experiments/run_pc_routes.py']
    report = dict(status='preparing', stage='export_calibration', results=[], prepared={}, pilots=[],
        expected_full_sweeps=expected_blocks(), expected_settings=24, repeats=5, batch_size=1,
        seeds=list(SEEDS), budgets=list(BUDGETS), models=names, baseline=baseline, resident_capacity=2,
        provenance=provenance, protected_hashes=protected, source_hashes={str(p):digest(p) for p in dependencies},
        validation_rgb_sha256=digest(cache_dir/'validation_rgb128.npy'),
        timing_boundary='Historical RGB128 measure(): transfer/normalization, fresh TRT router, host selection, capacity2 lazy hash/load/eviction, classifier inference, synchronization/host predictions. Decode/resize/pinning excluded.',
        energy_scope='Gross NVML GPU energy including idle/display; excludes CPU/rest of PC',
        protocol='All choices frozen from validation before opening test data. No retraining. Pilot 256 images/seed for runtime only; no policy changes. Full 24cases x5trials x3roles; alternating role order, independent cold caches. No uniform random unless separately added.',
        created_utc=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(0))
    REPORTS.mkdir(parents=True,exist_ok=True)
    path = REPORTS/'study.json'
    def save():
        report['updated_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(path,report)
    save()
    def completed(identity, **values):
        report['results'].append(dict(id=identity,router='correctness_b1_stage',status='complete',**values))
        save()
        thermal()
    try:
        validation = np.load(cache_dir/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
        if validation.shape != (5000,3,128,128) or validation.dtype != np.uint8:
            raise RuntimeError('Validation RGB identity mismatch')
        buffers=SharedBatchInputs()
        for seed in SEEDS:
            thermal()
            marker_path=PIPELINE/f'artifacts/pc_routers_correctness/global_correctness_seed{seed}.json'
            old=json.loads(marker_path.read_text())
            fit,calibration=split_validation(validation_labels,seed)
            if (old['status']!='complete' or old['models']!=names or old['provenance']!=provenance
                or old['fit_positions']!=fit.tolist() or old['calibration_positions']!=calibration.tolist()
                or old['validation_rgb_sha256']!=report['validation_rgb_sha256'] or len(old['training_loss'])!=20
                or digest(Path(old['weights']))!=old['weights_sha256']):
                raise RuntimeError('Historical weights/split/provenance invalid')
            model=CorrectnessRouter(213).cuda()
            model.load_state_dict(torch.load(old['weights'],map_location='cuda',weights_only=True))
            wrapped=BatchCorrectnessRouter(model).eval()
            engine,error=build_router(wrapped,'normalized_images',ARTIFACTS/'engines',f'correctness_seed{seed}_b1',input_shape=(1,3,128,128))
            if error>1e-5:
                raise RuntimeError('Router random-input parity exceeds historical bound')
            runner=RouterRunner(engine)
            if runner.shape!=(1,3,128,128) or tuple(runner.output.shape)!=(1,213):
                raise RuntimeError('Actual engine input/output shape invalid')
            q=[]
            real_error=0.
            with torch.inference_mode():
                for number,position in enumerate(calibration):
                    if number%100==0:
                        thermal()
                    inputs=buffers.prepare(torch.from_numpy(np.array(validation[position:position+1])).pin_memory())
                    actual=runner.run(inputs.contiguous()).cpu().numpy().copy()[0]
                    if number<100:
                        expected=wrapped(inputs).cpu().numpy()[0]
                        real_error=max(real_error,float(np.max(np.abs(actual-expected))))
                        if not np.allclose(actual,expected,atol=1e-5,rtol=1e-4):
                            raise RuntimeError('Actual validation-input parity failure')
                    q.append(actual)
            q=np.asarray(q)
            counts=correct[calibration].astype(int)
            policies={}
            comparators={}
            for objective in ('latency','energy'):
                costs=costs_for(records,1,objective)
                policies[objective]=dict(costs=costs.tolist(), budgets={str(b):calibrate_correctness(q,counts,np.ones(len(calibration)),costs,base_index,b) for b in BUDGETS})
                comparators[objective]={str(b):eligible_static(records,calibration,validation_labels,static_dir,by_name[baseline],1,objective,b) for b in BUDGETS}
            prepared=dict(seed=seed,engine=str(engine),engine_sha256=digest(engine),
                torch_max_error=error,validation_max_error=real_error,weights_sha256=old['weights_sha256'],
                marker_sha256=digest(marker_path),fit_positions=fit.tolist(),calibration_positions=calibration.tolist(),
                policies=policies,comparators=comparators,calibration_images=2000)
            report['prepared'][str(seed)]=prepared
            atomic_json(ARTIFACTS/f'prepared_seed{seed}.json',prepared)
            completed(f'prepare_seed{seed}',kind='preparation',seed=seed)
            print(f'PREPARED seed={seed} random_parity={error:.3g} real_parity={real_error:.3g}',flush=True)
            del model,wrapped,runner
            gc.collect();torch.cuda.empty_cache()
        # All policies/comparators now frozen; only now open held-out inputs.
        report['stage']='pilot';save()
        images=np.load(cache_dir/'test_rgb128.npy',mmap_mode='r',allow_pickle=False)
        labels=np.load(cache_dir/'test_labels.npy',allow_pickle=False)
        if images.shape!=(10000,3,128,128) or len(labels)!=10000:
            raise RuntimeError('Held-out input identity mismatch')
        report['test_hashes']={str(cache_dir/name):digest(cache_dir/name) for name in ('test_rgb128.npy','test_labels.npy')}
        def release():
            gc.collect();torch.cuda.empty_cache()
        def factory(record):
            if digest(Path(record['engine_path']))!=record['engine_sha256']:
                raise RuntimeError('Classifier engine hash mismatch')
            return EngineRunner(record['engine_path'],maximum_batch=1)
        def sweep(data,target,batches,metadata,objective,budget,role):
            thermal()
            if role=='learned':
                engines=EngineCache(records,2,factory,torch.cuda.synchronize,release)
                routers={1:RouterRunner(metadata['engine'])}
                policy=dict(metadata['policies'][objective]['budgets'][str(budget)],costs=metadata['policies'][objective]['costs'])
                chosen=baseline
            else:
                chosen=baseline if role=='reference' else metadata['comparators'][objective][str(budget)]
                engines=EngineCache([by_name[chosen]],1,factory,torch.cuda.synchronize,release)
                routers=policy=None
            try:
                value=measure(data,target,batches,engines,base_index if role=='learned' else 0,routers,policy,
                    'correctness_images',repeats=1,warmup=3,minimum_seconds=0)[0]
                validate_meter(value,len(target))
                return value,chosen
            finally:
                engines.clear()
                del routers
                release()
        for seed in SEEDS:
            metadata=report['prepared'][str(seed)]
            positions=np.random.default_rng(seed+1).permutation(len(labels))[:256]
            value,_=sweep(np.array(images[positions]),labels[positions],make_batches(np.arange(256),1,seed+1),metadata,'latency',1.,'learned')
            pilot=dict(seed=seed,images=256,measurement=value,extrapolated_learned_sweep_seconds=value['rgb_input_to_prediction_seconds']*10000/256)
            report['pilots'].append(pilot)
            completed(f'pilot_seed{seed}',kind='pilot',seed=seed)
            print(f"PILOT seed={seed} estimated full learned sweep={pilot['extrapolated_learned_sweep_seconds']:.1f}s",flush=True)
        report['stage']='full';report['status']='running';save()
        for seed in SEEDS:
            metadata=report['prepared'][str(seed)]
            batches=make_batches(np.arange(len(labels)),1,seed+1)
            for objective in ('latency','energy'):
                for budget in BUDGETS:
                    for trial in range(5):
                        roles=('reference','eligible_static','learned') if trial%2==0 else ('learned','eligible_static','reference')
                        for role in roles:
                            value,chosen=sweep(images,labels,batches,metadata,objective,budget,role)
                            value['repeat']=trial
                            completed(f'{objective}_seed{seed}_loss{budget}_trial{trial}_{role}',kind='full',seed=seed,
                                objective=objective,budget_pp=budget,trial=trial,role=role,chosen_static=chosen,measurement=value)
                            count=sum(r['kind']=='full' for r in report['results'])
                            print(f'FULL {count}/{expected_blocks()} seed={seed} {objective} allowance={budget} trial={trial} {role}',flush=True)
        if any(digest(Path(p))!=h for p,h in protected.items()):
            raise RuntimeError('Protected historical file changed')
        report['status']='complete';report['stage']='measurements_complete';report['protected_unchanged']=True
        save()
        print('COMPLETE 360 full sweeps; run analysis/summarize_correctness_b1.py with bundled Python',flush=True)
    except Exception as error:
        report['status']='failed';report['error']=repr(error);save()
        raise


if __name__=='__main__':
    main()
