"""Fixed-count paired timing, aggregatecoldenergy and actual FP32TRTselector."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import gc,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import select_correctness

REPORT=PIPELINE/'reports/dispatch_controls_20261029/study.json'
ARTIFACTS=PIPELINE/'artifacts/dispatch_controls_20261029'

def main():
    if REPORT.exists():raise RuntimeError('Existing controls preserved; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt,TRT_LOGGER
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner
    from pc_tests.model_pipeline.runtime.compact_switching import VerifiedHostStore
    from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    from pc_tests.model_pipeline.routers.correctness.device_selection import DeviceCostSelection
    torch.set_num_threads(1)
    predecessor=PIPELINE/'reports/batch8_grouping_20261029/audited_summary.json';assert json.loads(predecessor.read_text())['audit_passed']
    original=json.loads((PIPELINE/'reports/compact_switching_confirmation_20261028/study.json').read_text());pool=original['pool']
    oldplan=json.loads((PIPELINE/'artifacts/compact_switching_confirmation_20261028/frozen_plan.json').read_text());policy=next(p['static'] for p in oldplan['plans'] if p['seed']==42)
    costs=np.array(oldplan['costs_seconds']);cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,_,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    names=[r['name'] for r in records];columns=[names.index(n) for n in pool];byname={r['name']:r for r in records};records=[byname[n] for n in pool]
    _,cal=split_validation(labels,42);images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    host=[torch.from_numpy(np.array(images[p:p+1])).pin_memory() for p in cal[:512]]
    report=dict(status='running',results=[],source_sha256=digest(Path(__file__)),test_data_opened=False,provenance=provenance,pool=pool,seed=42,
        predecessor_sha256=digest(predecessor),policy=policy,costs_seconds=costs.tolist(),expected_rows=21,
        protocol='Validation-onlyengineering:5fixed512image sameclassifier A/Bpairedrepeats in16imageinterleavedblocks;3repeats x3selectors512images withrealCNN/selectedclassifier;3pairedfile/hostcoldload+cleanupwindows>=2s. No testlabels/parameters tuned. Original6engines; finalhostclass. FP32TRTselection separateapproximation,near-tie differences retained,not silentlyequatedtoFP64. GPUenergy excludesCPU.',created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='dispatch_controls',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report);meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVMLenergy unavailable')
    def telemetry():
        n=meter.nvml;h=meter.handle;out={'utc':datetime.now(timezone.utc).isoformat()}
        for name,fn in [('temperature_c',lambda:n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU)),('sm_clock_mhz',lambda:n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_SM)),('memory_clock_mhz',lambda:n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_MEM)),('power_w',lambda:n.nvmlDeviceGetPowerUsage(h)/1000),('pstate',lambda:n.nvmlDeviceGetPerformanceState(h)),('throttle_reasons',lambda:n.nvmlDeviceGetCurrentClocksThrottleReasons(h))]:
            try:out[name]=fn()
            except Exception as exc:out[name]={'unavailable':str(exc)}
        return out
    def cleanup():gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    buffers=SharedBatchInputs(maximum_batch=1)
    try:
      with torch.inference_mode():
        # Read-only sensoravailability; power snapshots are not cumulativeenergy.
        command="[Console]::OutputEncoding=[Text.Encoding]::UTF8; $result=@(); foreach($ns in @('root/LibreHardwareMonitor','root/OpenHardwareMonitor')) { try { $s=@(Get-CimInstance -Namespace $ns -ClassName Sensor -ErrorAction Stop | Where-Object { $_.SensorType -match 'Power|Energy' -and $_.Name -match 'CPU|Package|Total' } | Select-Object Name,SensorType,Value); $result+=@{namespace=$ns;sensors=$s;available=$true} } catch { $result+=@{namespace=$ns;available=$false;reason=$_.Exception.Message} } }; $result|ConvertTo-Json -Depth 5 -Compress"
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,encoding='utf-8',errors='replace',check=True)
        report['cpu_system_sensor_availability']=json.loads(result.stdout.strip().lstrip('\ufeff'))
        report['cpu_energy_measured']=False;report['system_energy_measured']=False
        report['sensor_scope']='Onlyavailability queried; instantaneouspower alone doesnotvalidate CPUtrainingenergy/payback. No driverinstallation,powerchange,smartplug or externaldriveaccess.'
        store=VerifiedHostStore(records);router=RouterRunner(PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine')
        engines=[create_runner(r,verified_bytes=store.get(r['name'])) for r in records]
        fixed=pool.index('resnet101__p0__fp16')
        for repeat in range(5):
            check(__file__);blocks=[]
            for first in range(0,512,16):
                pair={}
                for mode in (('static','forced_routed') if (repeat+first//16)%2==0 else ('forced_routed','static')):
                    meter.start();start=time.perf_counter();pred=[]
                    for h in host[first:first+16]:
                        normalized=buffers.prepare(h)
                        if mode=='forced_routed':
                            q=router.run(normalized).cpu().numpy().copy()[:,columns]
                            select_correctness(q,costs,policy['penalty']) # deliberatelyignored modelchoice, sameclassifier control
                        pred.append(int(engines[fixed].run(normalized,copy_input=False).argmax(1).item()))
                    torch.cuda.synchronize();pair[mode]=dict(seconds=time.perf_counter()-start,energy=meter.finish(16),predictions=pred)
                if pair['static']['predictions']!=pair['forced_routed']['predictions']:raise RuntimeError('Sameclassifier pairedprediction mismatch')
                blocks.append(pair)
                if first%128==0:check(__file__)
            done(f'paired_{repeat}',kind='same_classifier_pair',repeat=repeat,blocks=blocks,telemetry=telemetry(),images_per_mode=512,
                 note='ForcedsameR101FP16engine/context, actualCNNprobabilityreadback/costcalculation thenoverride. Not deployablepolicy oraccuracyclaim.')
        class Choice(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('columns',torch.tensor(columns,dtype=torch.int64));self.register_buffer('costs',torch.tensor(costs/costs.max(),dtype=torch.float32))
            def forward(self,q):return (q.index_select(1,self.columns)-float(policy['penalty'])*self.costs).argmax(1).to(torch.int32)
        model=Choice().cuda().eval();graph=ARTIFACTS/'selector_fp32.onnx';enginepath=ARTIFACTS/'selector_fp32.engine'
        torch.onnx.export(model,torch.zeros((1,213),device='cuda'),graph,input_names=['probabilities'],output_names=['choice'],opset_version=17,dynamo=False)
        builder=trt.Builder(TRT_LOGGER);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,TRT_LOGGER)
        if not parser.parse(graph.read_bytes()):raise RuntimeError('SelectorONNXparse failed: '+';'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,64*1024**2);config.clear_flag(trt.BuilderFlag.TF32)
        value=builder.build_serialized_network(network,config)
        if value is None:raise RuntimeError('SelectorTRTbuild failed')
        enginepath.write_bytes(bytes(value));selector=FixedGPURunner(enginepath);selector64=DeviceCostSelection(costs,policy['penalty']).cuda()
        def index(q,mode):
            if mode=='cpu64':return int(select_correctness(q.cpu().numpy().copy()[:,columns],costs,policy['penalty'])[0])
            if mode=='torch64':return int(selector64(q.index_select(1,model.columns)).item())
            return int(selector.run(q)['choice'].item())
        real_mismatch=0;near_mismatch=0;near_count=0
        for h in host:
            q=router.run(buffers.prepare(h));expected=index(q,'cpu64')
            if index(q,'torch64')!=expected:raise RuntimeError('FP64GPU selector changedchoice')
            real_mismatch+=index(q,'trt32')!=expected
        for a in range(6):
          for b in range(a+1,6):
           for delta in (-1e-6,-1e-7,-1e-8,0.,1e-8,1e-7,1e-6):
            q=np.zeros((1,213),dtype=np.float32);q[0,columns[a]]=.5+policy['penalty']*costs[a]/costs.max();q[0,columns[b]]=.5+policy['penalty']*costs[b]/costs.max()+delta
            tensor=torch.from_numpy(q).cuda();near_mismatch+=index(tensor,'trt32')!=index(tensor,'cpu64');near_count+=1
        done('selector_parity',kind='selector_parity',real_images=512,fp32_real_choice_mismatches=int(real_mismatch),near_tie_fixtures=near_count,
             near_tie_choice_mismatches=int(near_mismatch),engine_sha256=digest(enginepath),graph_sha256=digest(graph))
        for repeat in range(3):
          modes=('cpu64','torch64','trt32');modes=modes[repeat:]+modes[:repeat]
          for mode in modes:
            check(__file__);counts=np.zeros(6,dtype=np.int64);pred=[];before=telemetry();meter.start();start=time.perf_counter()
            for h in host:
                normalized=buffers.prepare(h);q=router.run(normalized);chosen=index(q,mode)
                pred.append(int(engines[chosen].run(normalized,copy_input=False).argmax(1).item()));counts[chosen]+=1
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(512)
            if energy.get('gpu_energy_j') is None:raise RuntimeError('Selectorwindowenergy unavailable')
            done(f'selector_{repeat}_{mode}',kind='selector_window',repeat=repeat,mode=mode,images=512,seconds=elapsed,ms_per_image=1000*elapsed/512,
                 gpu_energy=energy,predictions=pred,selected_images=counts.tolist(),telemetry_before=before,telemetry_after=telemetry())
        del engines;cleanup()
        record=records[fixed]
        for repeat in range(3):
          for mode in (('file','host') if repeat%2==0 else ('host','file')):
            check(__file__);cleanup();meter.start();start=time.perf_counter();loads=0;minimumfree=2**63
            while time.perf_counter()-start<2.:
                runner=create_runner(record,verified_bytes=store.get(record['name']) if mode=='host' else None);torch.cuda.synchronize()
                free,_=torch.cuda.mem_get_info();minimumfree=min(minimumfree,free)
                if free<1024**3:raise RuntimeError('Coldloadreserve violated')
                del runner;cleanup();loads+=1
            elapsed=time.perf_counter()-start;energy=meter.finish(loads)
            if energy.get('gpu_energy_j') is None:raise RuntimeError('Aggregatecoldenergy unavailable')
            done(f'cold_{repeat}_{mode}',kind='aggregate_cold',repeat=repeat,mode=mode,loads=loads,seconds=elapsed,gpu_energy=energy,gpu_j_per_load=energy['gpu_energy_j']/loads,measurement_unit='load_context_cleanup_cycle',minimum_free_bytes=minimumfree,
                 telemetry=telemetry(),scope='Repeatedload/context/releasecleanupwindow>=2s, GPUenergy inclidlewhileCPUhash/deserializes. Perloadamortizedestimate, notindividualstartup energy; CPUenergyexcluded.')
        assert len(report['results'])==21
        report['expected_rows']=21;report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
