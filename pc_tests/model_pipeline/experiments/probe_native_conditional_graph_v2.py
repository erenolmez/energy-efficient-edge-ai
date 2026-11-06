"""Bounded correctness-only explicit CUDA conditional graph probe."""
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import ctypes as C
import hashlib
import json
import os
import subprocess
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
import pc_tests.model_pipeline.experiments.compare_dispatch_residency_v2 as repeat

OUT=PIPELINE/'artifacts/native_conditional_graph_v2_20261104'
REPORT=OUT/'probe.json'


def main():
    if REPORT.exists():
        raise RuntimeError('Existing probe retained; no duplicate or blind restart')
    with (OUT/'run.lock').open('x') as f:
        json.dump(dict(pid=os.getpid(),utc=repeat.utc()),f)
    state=dict(status='preparing',pid=os.getpid(),utc=repeat.utc(),test_data_accessed=False,
               scope='Correctness/support ONLY. No latency or energy-speedup claim.',verified=[])
    atomic_json(REPORT,state)
    guard=None
    native=None
    handle=C.c_void_p()
    try:
        identity=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            f'Get-CimInstance Win32_Process -Filter "ProcessId = {os.getpid()}" | Select-Object ProcessId,ExecutablePath,CommandLine,@{{n="creation_unix_ms";e={{[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}}}} | ConvertTo-Json'],text=True)
        atomic_json(OUT/'worker_identity.json',json.loads(identity))
        # Reuse the independent passive guard, but route telemetry ONLY to the
        # new probe directory; neither v2 source nor v2 artifacts are modified.
        repeat.OUT=OUT
        guard=repeat.Guard()
        state['start_telemetry']=guard.start()
        import torch
        from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
        from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt
        torch.set_num_threads(1)
        original=PIPELINE/'reports/gpu_conditional_20261029/study.json'
        prior=json.loads(original.read_text())
        v2dir=PIPELINE/'artifacts/dispatch_residency_v2_20261104'
        v2=json.loads((v2dir/'study.json').read_text())
        assert v2['status']=='complete_pending_audit' and len(v2['rows'])==48
        for name,expected_hash in v2['reference_sha256'].items():
            assert repeat.digest(v2dir/name)==expected_hash
        ref_logits=np.load(v2dir/'reference_logits.npy',allow_pickle=False)
        ref_choices=np.load(v2dir/'reference_choices.npy',allow_pickle=False)
        ref_pred=np.load(v2dir/'reference_predictions.npy',allow_pickle=False)
        records=[r for r in prior['results'] if r['kind']=='branch_build']
        paths=[Path(r['engine']) for r in records]
        router_path=PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine'
        assert repeat.digest(router_path)==prior['router_engine_sha256']
        for p,r in zip(paths,records):
            assert repeat.digest(p)==r['engine_sha256']
        dll=OUT/'native_conditional_graph.dll'
        source=PIPELINE/'runtime/native_conditional_graph_v2.cu'
        cache=PIPELINE/'artifacts/pc_precision/data_cache/validation_rgb128.npy'
        protected=[Path(__file__),source,dll,original,router_path,*paths,cache,v2dir/'study.json',
            *(v2dir/name for name in v2['reference_sha256']),PIPELINE/'experiments/compare_dispatch_residency_v2.py']
        state['protected_hashes']={str(p):repeat.digest(p) for p in protected}
        state['revision']='v2: per-call CUDA diagnostics and pre-instantiation structure validation; old failed attempt preserved'
        state['toolchain']=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,tensorrt=trt.__version__,
            native_cuda='12.6',host_compiler='MSVC14.44',nvcc_allow_unsupported_compiler=True,
            note='Separate local helper only; no runtime installation/upgrades, no global environment changes. Mixed runtime interoperability tested, not assumed.')
        image=np.load(cache,mmap_mode='r',allow_pickle=False)
        host=torch.from_numpy(np.array(image[v2['input_positions']],copy=True)).float()
        host.div_(255).sub_(torch.tensor([.5071,.4867,.4408])[None,:,None,None])
        host.div_(torch.tensor([.2675,.2565,.2761])[None,:,None,None])
        assert repeat.array_hash(host.numpy())==v2['input_sha256']
        inputs=host.to('cuda')
        stage=torch.empty((1,3,128,128),device='cuda')
        stream=torch.cuda.Stream()
        router=FixedGPURunner(router_path)
        branches=[FixedGPURunner(p) for p in paths]
        for r in [router,*branches]:
            assert r.shape==(1,3,128,128)
            r.context.set_tensor_address(r.input,stage.data_ptr())
        prob=next(iter(router.outputs.values()))
        branch_outputs=[next(iter(r.outputs.values())) for r in branches]
        output=torch.empty((1,100),device='cuda')
        choice=torch.empty(1,device='cuda',dtype=torch.int32)
        counts=torch.zeros(2,device='cuda',dtype=torch.int32)
        prediction=torch.empty(1,device='cuda',dtype=torch.int32)
        assert torch.cuda.mem_get_info()[0]>=1024**3
        native=C.CDLL(str(dll))
        vp=C.c_void_p
        signatures={
            'graph_create': [C.POINTER(vp),vp,vp,vp,vp,vp,vp,vp,C.c_int,C.c_int,C.c_float,C.c_float,C.c_float],
            'graph_begin_router':[vp,vp], 'graph_end_router':[vp,vp],
            'graph_begin_body':[vp,vp,C.c_int], 'graph_end_body':[vp,vp,C.c_int],
            'graph_finish':[vp,C.c_char_p], 'graph_instantiate':[vp], 'graph_launch':[vp,vp],
            'graph_node_counts':[vp,C.POINTER(C.c_int)], 'graph_destroy':[vp]}
        for name,args in signatures.items():
            getattr(native,name).argtypes=args
            getattr(native,name).restype=C.c_int
        native.graph_last_error.restype=C.c_char_p
        def call(name,*args):
            status=getattr(native,name)(*args)
            if status:
                raise RuntimeError(name+': '+native.graph_last_error().decode(errors='replace'))
        # Recover exact original router column order from saved metadata only.
        from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation
        validation_records,_,_,_,_=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache.parent)
        names=[r['name'] for r in validation_records]
        ia,ib=[names.index(n) for n in prior['pair']]
        costs=np.array(prior['costs'],dtype=np.float32)
        offset=(float(prior['budget_policy']['penalty'])*costs/costs.max()).astype(np.float32)
        # Warm each fixed engine outside capture; all enqueues are checked.
        guard.phase='warmup'
        with torch.inference_mode(),torch.cuda.stream(stream):
            stage.copy_(inputs[:1])
            for _ in range(3):
                for r in [router,*branches]:
                    repeat.checked_enqueue(r.context,stream)
            stream.synchronize()
            call('graph_create',C.byref(handle),prob.data_ptr(),branch_outputs[0].data_ptr(),branch_outputs[1].data_ptr(),
                 output.data_ptr(),choice.data_ptr(),counts.data_ptr(),prediction.data_ptr(),ia,ib,
                 float(offset[0]),float(offset[1]),float(prior['engineering_bias']))
            guard.phase='graph_construction'
            state['status']='capturing_standalone_engines'
            atomic_json(REPORT,state)
            call('graph_begin_router',handle,stream.cuda_stream)
            try:
                repeat.checked_enqueue(router.context,stream)
            finally:
                call('graph_end_router',handle,stream.cuda_stream)
            for i in range(2):
                call('graph_begin_body',handle,stream.cuda_stream,i)
                try:
                    repeat.checked_enqueue(branches[i].context,stream)
                finally:
                    call('graph_end_body',handle,stream.cuda_stream,i)
            call('graph_finish',handle,str(OUT/'native_graph.dot').encode())
            node_counts=(C.c_int*9)()
            call('graph_node_counts',handle,node_counts)
            state['graph_node_counts']=dict(parent=list(node_counts[:3]),branchA=list(node_counts[3:6]),branchB=list(node_counts[6:9]),
                legend='[total nodes, host nodes, conditional nodes]')
            assert node_counts[1]==node_counts[4]==node_counts[7]==0, 'Host callbacks prohibited'
            assert node_counts[2]==2 and node_counts[3]>1 and node_counts[6]>1
            call('graph_instantiate',handle)
            state['instantiation_succeeded']=True
            state['status']='independent_graph_validation'
            atomic_json(REPORT,state)
            guard.cool()
            guard.phase='graph_correctness'
            maxerr=0
            # Two different input orders, no eager inference between poison and
            # replay. Validate every graph output AND selected-only counters.
            orders=[list(range(128)),np.random.default_rng(20261104).permutation(128).tolist()]
            for pass_id,order in enumerate(orders):
                for i in order:
                    guard.check()
                    stage.copy_(inputs[i:i+1])
                    output.fill_(float('nan'));choice.fill_(-1);prediction.fill_(-1);counts.zero_()
                    call('graph_launch',handle,stream.cuda_stream)
                    observed=output.cpu().numpy().copy()
                    actual_choice=int(choice.item());actual_prediction=int(prediction.item())
                    actual_counts=counts.cpu().numpy().copy()
                    expected_counts=np.zeros(2,dtype=np.int32);expected_counts[int(ref_choices[i])]=1
                    assert np.isfinite(observed).all(), 'Stale/unwritten graph output'
                    assert actual_choice==int(ref_choices[i]), 'Graph choice mismatch'
                    assert actual_prediction==int(ref_pred[i]), 'Graph prediction mismatch'
                    assert np.array_equal(actual_counts,expected_counts), 'Selected-only branch counter mismatch'
                    assert np.allclose(observed,ref_logits[i:i+1],atol=1e-3,rtol=1e-4), 'Graph logit mismatch'
                    maxerr=max(maxerr,float(np.abs(observed-ref_logits[i:i+1]).max()))
                    state['verified'].append(dict(pass_id=pass_id,input_index=i,choice=actual_choice,prediction=actual_prediction,branch_counts=actual_counts.tolist()))
                atomic_json(REPORT,state)
        for p,h in state['protected_hashes'].items():
            assert repeat.digest(p)==h, 'Protected data changed: '+p
        state.update(status='correctness_verified_pending_audit',max_abs_logit_error=maxerr,
            verified_replays=len(state['verified']),choice_mismatches=0,prediction_mismatches=0,
            stale_output_checks='NaN/-1 poisoning before every replay, no eager after poisoning',
            graph_mechanism='Captured standalone router -> GPU condition setter -> two complementary CUDA If nodes containing captured classifier branches -> GPU argmax. No CPU callbacks/selection.',
            finished_utc=repeat.utc())
        atomic_json(REPORT,state)
    except BaseException as exc:
        state.update(status='stopped_error',error=repr(exc),finished_utc=repeat.utc())
        atomic_json(REPORT,state)
        raise
    finally:
        if guard:
            guard.close()
        if native and handle.value:
            native.graph_destroy(handle)


if __name__=='__main__':
    main()
