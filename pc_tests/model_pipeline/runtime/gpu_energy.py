"""NVML GPU energy-counter measurements; these exclude the rest of the PC."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import time


def counter_metrics(start_mj, end_mj, seconds, images):
    if seconds <= 0 or images <= 0 or end_mj < start_mj:
        raise ValueError('Invalid energy measurement interval')
    joules = (end_mj - start_mj) / 1000.0
    return {'gpu_energy_j': joules, 'gpu_energy_j_per_image': joules / images,
            'gpu_average_watts': joules / seconds, 'energy_window_seconds': seconds,
            'energy_window_images': images, 'energy_method': 'NVML total energy counter difference',
            'energy_scope': 'Total reported GPU energy, including idle/display load; not whole-PC energy or Jetson VDD_IN'}


class EnergyCounter:
    def __init__(self):
        self.available, self.error = False, None
        try:
            import pynvml
            self.nvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def start(self):
        if self.available:
            try:
                self.start_mj = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.started = time.perf_counter()
            except Exception as exc:
                self.available, self.error = False, str(exc)

    def finish(self, images):
        if not self.available:
            return {'gpu_energy_j_per_image': None, 'energy_unavailable_reason': self.error}
        try:
            end_mj = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            elapsed = time.perf_counter() - self.started
            metrics = counter_metrics(self.start_mj, end_mj, elapsed, images)
            if metrics['gpu_energy_j'] == 0:
                raise RuntimeError('NVML energy counter did not advance')
            metrics['gpu_temperature_c'] = self.nvml.nvmlDeviceGetTemperature(
                self.handle, self.nvml.NVML_TEMPERATURE_GPU)
            return metrics
        except Exception as exc:
            return {'gpu_energy_j_per_image': None, 'energy_unavailable_reason': str(exc)}
