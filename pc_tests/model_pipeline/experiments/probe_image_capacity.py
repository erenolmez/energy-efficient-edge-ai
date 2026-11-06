"""Conservative RTX image residency probe using synthetic 128px inputs only."""
import gc
import json
import os
from pathlib import Path
import subprocess

from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.experiments.compare_router_warmup_dose import digest


def main():
    import pynvml as nv
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner

    out = PIPELINE/'artifacts/image_capacity_20261103'
    out.mkdir(exist_ok=False)
    reserve = 1024**3
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    def temperature():
        t = nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU)
        if t >= 85:
            raise RuntimeError('GPU temperature stop')
        return t
    if temperature() > 70:
        raise RuntimeError('GPU start temperature exceeds 70 C')
    cmd = f'Get-CimInstance Win32_Process -Filter "ProcessId = {os.getpid()}" | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress'
    identity = json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',cmd],text=True))
    report = dict(status='running',worker=identity,gpu=torch.cuda.get_device_name(0),reserve_bytes=reserve,
                  batch=128,rows=[],synthetic_only=True)
    atomic_json(out/'study.json',report)
    try:
        for loaded in (False,True):
            models = []
            router = None
            inputs = None
            if loaded:
                records = json.loads((PIPELINE/'reports/gpu_static.json').read_text())['results']
                marker = json.loads((PIPELINE/'artifacts/pc_routers_correctness/global_correctness_seed42.json').read_text())
                engine = marker['engines']['128']
                specs = [(engine['engine'],engine['sha256'])]
                for name in ('resnet34__p0__fp16','resnet101__p0__fp16'):
                    r = next(r for r in records if r['name']==name)
                    specs.append((r['engine_path'],r['engine_sha256']))
                for p,h in specs:
                    if Path(p).resolve().drive.upper()!='C:' or digest(p)!=h:
                        raise RuntimeError('Engine path or hash mismatch')
                router = RouterRunner(specs[0][0])
                models = [EngineRunner(p,maximum_batch=128) for p,_ in specs[1:]]
                inputs = torch.zeros((128,3,128,128),device='cuda')
                report['loaded_engines'] = specs
            def verify(x):
                if not loaded:
                    return
                for engine, width in [(router,213), *[(m,100) for m in models]]:
                    y = engine.run(x).cpu()
                    if y.shape != (1 if width==213 else 128,width) or not torch.isfinite(y).all():
                        raise RuntimeError('Inference verification failed')
                torch.cuda.synchronize()
            with torch.inference_mode():
                verify(inputs)
                for dtype in (torch.uint8,torch.float32):
                    chunks = []
                    torch.cuda.empty_cache()
                    start_free = torch.cuda.mem_get_info()[0]
                    chunk_bytes = 128*3*128*128*(1 if dtype==torch.uint8 else 4)
                    # Stop conservatively; never search for an OOM.
                    while torch.cuda.mem_get_info()[0] >= reserve + chunk_bytes + 64*1024**2:
                        temperature()
                        chunks.append(torch.zeros((128,3,128,128),dtype=dtype,device='cuda'))
                        torch.cuda.synchronize()
                    if not chunks:
                        raise RuntimeError('No safely admissible image chunk')
                    if loaded:
                        if dtype==torch.uint8:
                            inputs.copy_(chunks[-1]).div_(255)
                            for channel,mean,std in zip(inputs.unbind(1),(.5071,.4867,.4408),(.2675,.2565,.2761)):
                                channel.sub_(mean).div_(std)
                            verify(inputs)
                        else:
                            verify(chunks[-1])
                    free = torch.cuda.mem_get_info()[0]
                    if free < reserve:
                        raise RuntimeError('Reserve breached; do not report capacity as safe')
                    report['rows'].append(dict(models_loaded=loaded,dtype=str(dtype),images=128*len(chunks),
                        image_bytes=chunk_bytes*len(chunks),start_free_bytes=start_free,final_free_bytes=free,
                        inference_verified=loaded,gpu_c=temperature()))
                    atomic_json(out/'study.json',report)
                    chunks.clear()
                    gc.collect()
                    torch.cuda.empty_cache()
        report['status']='complete'
    except Exception as exc:
        report.update(status='stopped',error=repr(exc))
        raise
    finally:
        atomic_json(out/'study.json',report)


if __name__=='__main__':
    main()
