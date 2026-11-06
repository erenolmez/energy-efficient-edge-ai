"""Validation-only TensorRT FP16 screen for the merged fine-tuned bank."""
import argparse
import csv
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as telemetry
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner, build_engine, trt
from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs


BATCHES = (1, 8, 16, 32, 64, 128)
CACHE = PIPELINE / 'artifacts/pc_precision/data_cache'
PC_BANK = PIPELINE / 'reports/fresh_prune_adam_bank_20261105'
RESERVE = 1024**3


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_bank(index):
    with index.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    plan = read(PC_BANK / 'plan.json')
    pc = {r['id']: r for r in read(PC_BANK / 'campaign.json')['completed']}
    cases = {r['id']: r for r in plan['cases']}
    require(len(rows) == len(cases) == 63 and {r['case'] for r in rows} == set(cases), 'Incomplete model bank')
    verified = {}
    for row in rows:
        name, owner = row['case'], row['owner']
        path = index.parent / name / 'best.pth'
        require(path.resolve() == Path(row['checkpoint']).resolve(), f'Unexpected index path for {name}')
        if owner == 'pc-original':
            digest = pc[name]['best']['sha256']
            require(float(row['best_validation_accuracy']) == pc[name]['best']['accuracy'], 'PC accuracy changed')
        else:
            require(re.fullmatch(r'hpc[0-2]', owner), 'Unknown checkpoint owner')
            receipt = read(path.parent / 'receipt.json')
            require(receipt['id'] == name and receipt['owner'] == owner, 'Wrong imported receipt')
            digest = receipt['checkpoint_sha256']
        require(telemetry.digest(path) == digest, f'Checkpoint hash mismatch for {name}')
        verified[name] = dict(case=cases[name], checkpoint=str(path), checkpoint_sha256=digest, owner=owner,
                              training_best_validation_accuracy=float(row['best_validation_accuracy']))
    return verified, plan['split_sha256']


def checked(guard):
    guard.check()
    require(torch.cuda.mem_get_info()[0] >= RESERVE, 'Below 1 GiB GPU memory reserve')


def cool(guard, threshold=68):
    deadline = time.monotonic() + 180
    while guard.last['temperature_c'] > threshold or guard.last['clock_reasons'] & 96:
        checked(guard)
        require(time.monotonic() < deadline, 'GPU did not cool into the measurement band')
        time.sleep(.5)


def windows(runner, images, size, guard, repeats=3):
    host = torch.from_numpy(np.array(images[:size])).pin_memory()
    inputs = SharedBatchInputs(size)
    values = []
    for repeat in range(repeats):
        cool(guard)
        for _ in range(20):
            runner.run(inputs.prepare(host), copy_input=False).argmax(1).cpu()
        torch.cuda.synchronize()
        token = f'static:{guard.phase.split(":")[1]}:b{size}:r{repeat}'
        guard.phase = token
        energy = EnergyCounter()
        require(energy.available, 'GPU energy counter unavailable')
        energy.start()
        begin = time.perf_counter()
        count = 0
        while count < 30 or time.perf_counter() - begin < 3:
            runner.run(inputs.prepare(host), copy_input=False).argmax(1).cpu()
            count += 1
            if count % 32 == 0:
                checked(guard)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        e = energy.finish(count * size)
        samples = [x for x in guard.samples if x['phase'] == token]
        checked(guard)
        require(samples and not any(x['clock_reasons'] & 96 for x in samples) and
                not guard.last['clock_reasons'] & 96, f'Thermally qualified window failed: {token}')
        require(e.get('gpu_energy_j') is not None, 'Missing GPU energy measurement')
        values.append(dict(repeat=repeat, batches=count, seconds=elapsed, ms_batch=1000*elapsed/count,
                           mj_batch=1000*e['gpu_energy_j']/count, telemetry_samples=len(samples),
                           peak_c=max(x['temperature_c'] for x in samples),
                           power_cap_samples=sum(bool(x['clock_reasons'] & 4) for x in samples)))
        guard.phase = f'static:{token.split(":")[1]}:cooldown'
    return values


def evaluate(runner, images, labels, guard):
    predictions = []
    for offset in range(0, len(labels), 32):
        x = torch.from_numpy(np.array(images[offset:offset+32])).cuda()
        out = runner.run(x, copy_input=False).cpu().numpy()
        require(out.shape == (len(x), 100) and np.isfinite(out).all(), 'Invalid engine output')
        predictions.extend(out.argmax(1))
        checked(guard)
    return np.asarray(predictions, dtype=np.int16)


def export_onnx(out, name, row):
    directory = out / 'graphs'
    directory.mkdir(parents=True, exist_ok=True)
    graph = directory / f'{name}.onnx'
    receipt = directory / f'{name}.json'
    if receipt.is_file():
        old = read(receipt)
        require(old['checkpoint_sha256'] == row['checkpoint_sha256'] and
                old['onnx_sha256'] == telemetry.digest(graph), f'Existing ONNX provenance differs for {name}')
        return graph
    require(not graph.exists(), f'Existing unaudited graph for {name}; preserve and inspect')
    pending = graph.with_suffix('.pending.onnx')
    require(not pending.exists(), f'Interrupted export for {name}; preserve and inspect')
    model = torch.load(row['checkpoint'], map_location='cpu', weights_only=False).eval()
    torch.onnx.export(model, (torch.zeros(1,3,128,128),), pending, opset_version=17,
                      input_names=['input'], output_names=['logits'], dynamo=False,
                      dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}})
    del model
    pending.rename(graph)
    atomic_json(receipt, dict(checkpoint_sha256=row['checkpoint_sha256'], onnx_sha256=telemetry.digest(graph),
                              opset=17, torch=torch.__version__))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='Verify checkpoints and export on CPU; no CUDA work')
    a = parser.parse_args()
    torch.set_num_threads(2)
    bank, split = verified_bank(a.index)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids = ['resnet18__p10', 'resnet50__p50'] if a.pilot else sorted(bank)
    if a.export_only:
        for n, name in enumerate(ids, 1):
            export_onnx(out, name, bank[name])
            print(f'Verified ONNX export {n}/{len(ids)}: {name}', flush=True)
        return
    require(read(CACHE/'router_data.json')['split_sha256'] == split, 'Wrong validation cache split')
    images = np.load(CACHE/'validation.npy', mmap_mode='r', allow_pickle=False)
    rgb = np.load(CACHE/'validation_rgb128.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(CACHE/'validation_labels.npy', allow_pickle=False)
    require(images.shape == rgb.shape == (5000,3,128,128) and images.dtype == np.float32 and rgb.dtype == np.uint8 and
            np.array_equal(np.bincount(labels), np.full(100, 50)), 'Unexpected validation data')
    run_id = time.strftime('%Y%m%d_%H%M%S')
    logdir = out / ('screen_' + run_id)
    logdir.mkdir()
    telemetry.OUT = logdir
    guard = telemetry.Guard()
    status = dict(status='running', expected=len(ids), completed=[], split_sha256=split,
                  boundary='prepared host RGB128 through GPU normalization to host predictions', energy='gross NVML GPU only',
                  validation_images=5000, precision='TensorRT FP16', batches=list(BATCHES))
    atomic_json(logdir/'status.json', status)
    try:
        guard.start()
        for name in ids:
            row = bank[name]
            folder = out/'models'/name
            if (folder/'result.json').is_file():
                previous = read(folder/'result.json')
                require(previous['checkpoint_sha256'] == row['checkpoint_sha256'] and
                        previous['split_sha256'] == split and previous['tensorrt'] == trt.__version__ and
                        telemetry.digest(folder/'model.engine') == previous['engine_sha256'] and
                        telemetry.digest(folder/'validation_predictions.npy') == previous['predictions_sha256'],
                        f'Existing result disagrees for {name}')
            else:
                folder.mkdir(parents=True, exist_ok=False)
                engine = folder/'model.engine'
                guard.phase = f'export:{name}'
                onnx = export_onnx(out, name, row)
                cool(guard)
                guard.phase = f'build:{name}'
                build_engine(onnx, engine, 'fp16')
                checked(guard)
                guard.phase = f'validation:{name}'
                runner = EngineRunner(engine)
                predictions = evaluate(runner, images, labels, guard)
                prediction_path = folder/'validation_predictions.npy'
                np.save(prediction_path, predictions)
                guard.phase = f'static:{name}:setup'
                measurements = {str(b): windows(runner, rgb, b, guard) for b in BATCHES}
                atomic_json(folder/'result.json', dict(**row, split_sha256=split, tensorrt=trt.__version__,
                            engine_sha256=telemetry.digest(engine), onnx_sha256=telemetry.digest(onnx),
                            predictions_sha256=telemetry.digest(prediction_path), timings=measurements,
                            validation_accuracy=100*float(np.mean(predictions == labels))))
                del runner
                gc.collect()
                torch.cuda.empty_cache()
            status['completed'].append(name)
            atomic_json(logdir/'status.json', status)
            print(f'Verified screen {len(status["completed"])}/{len(ids)}: {name}', flush=True)
        status['status'] = 'complete'
        atomic_json(logdir/'status.json', status)
    except BaseException as exc:
        status.update(status='stopped', error=repr(exc))
        atomic_json(logdir/'status.json', status)
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    main()
