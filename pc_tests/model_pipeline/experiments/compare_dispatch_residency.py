"""New, bounded batch-one dispatch controls. Never resumes an existing study."""
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import hashlib
import itertools
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE / 'artifacts/dispatch_residency_20261104'
REPORT = OUT / 'study.json'
PRIOR = PIPELINE / 'reports/gpu_conditional_20261029/study.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


class Safety:
    """Passive one-second telemetry; latch on missing temperature or >=85 C."""
    def __init__(self):
        import pynvml as nv
        nv.nvmlInit()
        self.nv, self.handle = nv, nv.nvmlDeviceGetHandleByIndex(0)
        self.stop = threading.Event()
        self.latch = None
        self.phase = 'setup'
        self.last_sample = 0
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        n, h = self.nv, self.handle
        temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
        reasons = int(n.nvmlDeviceGetCurrentClocksThrottleReasons(h))
        return dict(utc=utc(), phase=self.phase, temperature_c=int(temp),
                    clock_reasons=reasons,
                    graphics_clock_mhz=int(n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_GRAPHICS)),
                    memory_used_bytes=int(n.nvmlDeviceGetMemoryInfo(h).used))

    def start(self):
        first = self.sample()
        if first['temperature_c'] > 70 or first['clock_reasons'] & 32:
            raise RuntimeError('Start gate deferred: >70C or software thermal flag; no retry')
        self.last_sample = time.monotonic()
        self.thread.start()
        return first

    def loop(self):
        with (OUT / 'telemetry.jsonl').open('x', buffering=1) as f:
            while not self.stop.is_set():
                try:
                    row = self.sample()
                    self.last_sample = time.monotonic()
                    if row['temperature_c'] >= 85:
                        self.latch = 'Temperature reached 85C'
                except Exception as exc:
                    self.latch = 'Telemetry unavailable: ' + type(exc).__name__
                    row = dict(utc=utc(), phase=self.phase, error=self.latch)
                f.write(json.dumps(row) + '\n')
                self.stop.wait(1)

    def check(self):
        if self.latch or time.monotonic() - self.last_sample > 5:
            raise RuntimeError(self.latch or 'Telemetry stale')
        pause = PIPELINE / 'reports/experiment_pause.json'
        if json.loads(pause.read_text())['status'] == 'paused_by_user':
            raise RuntimeError('User pause requested')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


def main():
    if REPORT.exists():
        raise RuntimeError('Existing study preserved; no restart or duplicate')
    OUT.mkdir(parents=True, exist_ok=True)
    reservation = (OUT / 'run.lock').open('x')
    reservation.write(json.dumps(dict(pid=os.getpid(), started_utc=utc())))
    reservation.close()
    state = dict(status='preparing', started_utc=utc(), pid=os.getpid(), rows=[],
                 stage='batch_one_two_classifier_followup', test_data_accessed=False,
                 no_training=True, source_sha256=digest(__file__))
    atomic_json(REPORT, state)
    safety = None
    try:
        identity = subprocess.check_output(
            ['powershell.exe', '-NoProfile', '-Command',
             f'Get-CimInstance Win32_Process -Filter "ProcessId = {os.getpid()}" | '
             'Select-Object ProcessId,ExecutablePath,CommandLine,@{n="creation_unix_ms";e={[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}} | ConvertTo-Json'],
            text=True)
        atomic_json(OUT / 'worker_identity.json', json.loads(identity))
        # This study has its own guard. Do not rearm legacy experiments/guards.
        safety = Safety()
        state['start_telemetry'] = safety.start()
        import torch
        from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
        from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt
        from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
        torch.set_num_threads(1)
        prior = json.loads(PRIOR.read_text())
        assert prior['status'] == 'complete' and prior['branch_execution_verified']
        assert prior['parity']['cpu_vs_gpu_choice_mismatches'] == 0
        rows = prior['results']
        branches = [r for r in rows if r['kind'] == 'branch_build']
        combined_record = next(r for r in rows if r['kind'] == 'conditional_build')
        engine_paths = [Path(r['engine']) for r in branches] + [Path(combined_record['engine'])]
        for record, path in zip(branches + [combined_record], engine_paths):
            assert digest(path) == record['engine_sha256'], str(path)
        router_path = PIPELINE / 'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine'
        assert digest(router_path) == prior['router_engine_sha256']
        cache = PIPELINE / 'artifacts/pc_precision/data_cache/validation_rgb128.npy'
        protected = [PRIOR, cache, router_path, *engine_paths,
                     PIPELINE / 'runtime/fixed_gpu_runner.py',
                     PIPELINE / 'runtime/gpu_engine_preflight.py', Path(__file__)]
        state['protected_hashes'] = {str(p): digest(p) for p in protected}
        state['predecessor_protocol'] = prior['engineering_policy_note']
        state['batch'] = 1
        state['models'] = prior['pair']
        state['device'] = torch.cuda.get_device_name()
        state['torch_version'] = torch.__version__
        state['tensorrt_version'] = trt.__version__
        state['policy'] = {k: prior[k] for k in ('engineering_bias', 'budget_policy', 'costs')}
        positions = prior['calibration_positions'][:128]
        rgb = np.load(cache, mmap_mode='r', allow_pickle=False)
        value = torch.from_numpy(np.array(rgb[positions], copy=True)).float()
        value.div_(255).sub_(torch.tensor([.5071,.4867,.4408])[None,:,None,None])
        value.div_(torch.tensor([.2675,.2565,.2761])[None,:,None,None])
        host = value.pin_memory()
        device = host.to('cuda')
        state['validation_positions'] = positions
        state['normalized_input_sha256'] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        stage = torch.empty((1, 3, 128, 128), device='cuda')
        runners = [FixedGPURunner(p) for p in engine_paths[:2]]
        router = FixedGPURunner(router_path)
        combined = FixedGPURunner(engine_paths[2])
        all_runners = [*runners, router, combined]
        stream = torch.cuda.Stream()
        # Use exactly the same stream/runner for CPU and GPU branches. No hidden
        # runner stream synchronizations. Bind a stable staging input once.
        for runner in all_runners:
            assert runner.shape == (1, 3, 128, 128)
            runner.context.set_tensor_address(runner.input, stage.data_ptr())
        def enqueue(runner):
            if not runner.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT enqueue failed')
            return runner.outputs
        def logits(runner):
            return next(iter(enqueue(runner).values()))
        gpu_pred = torch.empty(128, dtype=torch.int64, device='cuda')
        columns = []
        # Names/columns are explicitly recovered from validation metadata only.
        from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation
        records, _, _, _, _ = load_validation(PIPELINE / 'reports/gpu_static.json',
            PIPELINE / 'artifacts/gpu_static', cache.parent)
        names = [r['name'] for r in records]
        columns = [names.index(n) for n in prior['pair']]
        costs = np.array(prior['costs'], dtype=np.float32)
        offset = (float(prior['budget_policy']['penalty']) * costs / costs.max()).astype(np.float32)
        bias = np.float32(prior['engineering_bias'])
        def choose(q):
            scores = q[0, columns].astype(np.float32) - offset
            return 0 if np.float32(scores[0] - scores[1]) >= bias else 1
        safety.phase = 'correctness'
        choices, errors = [], []
        with torch.inference_mode(), torch.cuda.stream(stream):
            for i in range(128):
                safety.check()
                stage.copy_(device[i:i+1])
                q = logits(router).cpu().numpy().copy()
                selected = choose(q)
                expected = logits(runners[selected]).cpu().numpy().copy()
                result = enqueue(combined)
                observed = result['logits'].cpu().numpy().copy()
                actual = int(result['choice'].item())
                assert selected == actual, 'Choice mismatch'
                assert np.allclose(expected, observed, atol=1e-3, rtol=1e-4), 'Logit mismatch'
                assert expected.argmax() == observed.argmax(), 'Class mismatch'
                choices.append(actual)
                errors.append(float(np.abs(expected-observed).max()))
        assert set(choices) == {0, 1}, 'Both branches must be exercised'
        state['parity'] = dict(images=128, choice_mismatches=0, class_mismatches=0,
                               max_abs_logit_error=max(errors), branch_counts=np.bincount(choices).tolist())
        # Re-verify unselected branch does not execute before using timings.
        class Profile(trt.IProfiler):
            def __init__(self):
                trt.IProfiler.__init__(self)
                self.rows = []
            def report_layer_time(self, name, ms):
                self.rows.append(dict(name=name, ms=float(ms)))
        profile = Profile()
        combined.context.profiler = profile
        combined.context.enqueue_emits_profile = True
        traces = []
        with torch.inference_mode(), torch.cuda.stream(stream):
            for selected in (0, 1):
                profile.rows = []
                stage.copy_(device[choices.index(selected):choices.index(selected)+1])
                enqueue(combined)
                stream.synchronize()
                if not profile.rows:
                    combined.context.report_to_profiler()
                masses = {prefix: sum(r['ms'] for r in profile.rows if prefix in r['name']
                    and any(k in r['name'].lower() for k in ('conv', 'gemm', 'matmul')))
                    for prefix in ('A__', 'B__')}
                assert masses['A__' if selected == 0 else 'B__'] > 0
                assert masses['B__' if selected == 0 else 'A__'] <= 1e-7
                traces.append(dict(choice=selected, compute_ms=masses, layers=list(profile.rows)))
        state['branch_execution'] = traces
        # Fresh unprofiled context, same engine; never time profiler overhead.
        combined.context = combined.engine.create_execution_context()
        for name, tensor in combined.outputs.items():
            combined.context.set_tensor_address(name, tensor.data_ptr())
        combined.context.set_tensor_address(combined.input, stage.data_ptr())
        free, _ = torch.cuda.mem_get_info()
        if free < 1024**3:
            raise RuntimeError('1GiB reserve violated')
        state['free_bytes_after_setup'] = free
        atomic_json(REPORT, state)
        graph = None
        safety.phase = 'graph_support_probe'
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    enqueue(combined)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    # TensorRT native If keeps routing decision on device.
                    combined.context.execute_async_v3(stream.cuda_stream)
                graph.replay()
                stream.synchronize()
            state['cuda_graph_supported'] = True
        except Exception as exc:
            # No silent eager replacement for an unsupported graph mode.
            state['cuda_graph_supported'] = False
            state['cuda_graph_error'] = repr(exc)
            graph = None
            # A capture error may poison context; stop rather than time it.
            state['status'] = 'blocked_graph_probe'
            atomic_json(REPORT, state)
            raise RuntimeError('CUDA graph capture unsupported; preserve evidence, no automatic fallback') from exc
        modes = ['cpu_dispatch', 'gpu_conditional', 'gpu_graph', 'static_a', 'static_b']
        variants = list(itertools.product(modes, ['host', 'gpu'], ['per_image_cpu', 'deferred_gpu']))
        state['protocol'] = dict(repeats=6, variants=variants, target_seconds=3,
            warmup_requests=100, idle_seconds=2, batch=1, request_cycle=128,
            scope='Two FP32 resident classifiers; frozen median-biased engineering policy, NOT accuracy-budget deployment policy.',
            timing='Pre-normalized FP32 inputs; CPU normalization/setup excluded for ALL variants. Host includes H2D; GPU includes matched D2D staging. Deferred GPU output has no class readback inside timed loop; checksum copied only after timer. CPU dispatch necessarily reads router probabilities. CPU still submits requests; graph replay reduces submissions but is not autonomous CPU-free execution.',
            fairness='Same input cycle, engines, policy and final-output condition. Six reproducible rotated/reversed orders; no classifier training, calibration, test access or HPC changes.',
            energy='Gross NVML GPU energy incl. display/idle; NOT CPU/whole-PC energy.',
            larger_batch_stage='Not implemented by this batch-one experiment; do not call serial request counts batch8/32.')
        atomic_json(OUT / 'plan.json', state['protocol'])
        def request(mode, residence, egress, i):
            stage.copy_((host if residence == 'host' else device)[i:i+1], non_blocking=True)
            if mode == 'cpu_dispatch':
                selected = choose(logits(router).cpu().numpy())
                out = logits(runners[selected])
            elif mode in ('gpu_conditional', 'gpu_graph'):
                if mode == 'gpu_graph':
                    graph.replay()
                else:
                    enqueue(combined)
                out = combined.outputs['logits']
            else:
                out = logits(runners[0 if mode == 'static_a' else 1])
            pred = out.argmax(1)
            if egress == 'per_image_cpu':
                return int(pred.item())
            gpu_pred[i:i+1].copy_(pred)
            return None
        # Confirm graph outputs for ALL inputs, including both branches.
        with torch.inference_mode(), torch.cuda.stream(stream):
            for i in range(128):
                stage.copy_(device[i:i+1])
                expected = enqueue(combined)['logits'].clone()
                graph.replay()
                assert torch.allclose(expected, combined.outputs['logits'], atol=1e-3, rtol=1e-4)
        state['graph_parity_images'] = 128
        meter = EnergyCounter()
        if not meter.available:
            raise RuntimeError('NVML energy unavailable')
        state['status'] = 'measuring'
        state['expected_windows'] = len(variants) * 6
        with torch.inference_mode(), torch.cuda.stream(stream):
            for repeat in range(6):
                order = variants[repeat*3:] + variants[:repeat*3]
                if repeat % 2:
                    order = list(reversed(order))
                for mode, residence, egress in order:
                    safety.check()
                    safety.phase = 'warmup'
                    warm_started = time.perf_counter()
                    for i in range(100):
                        request(mode, residence, egress, i % 128)
                        if i % 32 == 0:
                            safety.check()
                    stream.synchronize()
                    warm_seconds = time.perf_counter() - warm_started
                    # Choose work outside the timing window, so GPU-deferred
                    # mode needs no per-cycle host/device synchronization.
                    cycles = max(1, min(64, int(np.ceil(3 / (warm_seconds * 128 / 100)))))
                    safety.phase = f'window:{repeat}:{mode}:{residence}:{egress}'
                    state['current'] = dict(repeat=repeat, mode=mode, residence=residence, egress=egress)
                    atomic_json(REPORT, state)
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    stream.synchronize()
                    meter.start()
                    start = time.perf_counter()
                    start_event.record(stream)
                    count, cpu_checksum = 0, 0
                    for _ in range(cycles):
                        # A full input cycle per sweep gives identical branch mix.
                        for i in range(128):
                            value = request(mode, residence, egress, i)
                            if value is not None:
                                cpu_checksum += value
                            count += 1
                            if i % 32 == 0:
                                safety.check()
                    end_event.record(stream)
                    stream.synchronize()
                    seconds = time.perf_counter() - start
                    energy = meter.finish(count)
                    if not energy.get('gpu_energy_j') or not np.isfinite(energy['gpu_energy_j']):
                        raise RuntimeError('Nonpositive/unavailable GPU energy')
                    if egress == 'deferred_gpu':
                        cpu_checksum = int(gpu_pred.sum().item())
                    row = dict(repeat=repeat, mode=mode, residence=residence, egress=egress,
                               requests=count, seconds=seconds, ms_per_image=seconds*1000/count,
                               stream_span_ms=start_event.elapsed_time(end_event), gpu_energy=energy,
                               output_checksum=cpu_checksum, ended_utc=utc())
                    row['planned_cycles'] = cycles
                    row['timing_note'] = 'Fixed work from outside-window warmup estimate; final boundary sync only except intentional CPU-dispatch/per-image-egress readbacks'
                    state['rows'].append(row)
                    atomic_json(REPORT, state)
                    print(json.dumps({k: row[k] for k in ('repeat','mode','residence','egress','ms_per_image')}), flush=True)
                    safety.phase = 'idle'
                    time.sleep(2)
        for path, expected in state['protected_hashes'].items():
            assert digest(path) == expected, 'Protected input/source changed: ' + path
        state['status'] = 'complete_pending_audit'
        state['finished_utc'] = utc()
        atomic_json(REPORT, state)
    except BaseException as exc:
        state.update(status='stopped_error', error=repr(exc), stopped_utc=utc())
        atomic_json(REPORT, state)
        raise
    finally:
        if safety:
            safety.close()


if __name__ == '__main__':
    main()
