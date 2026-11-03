"""Real CUDA context check inside an allocated Slurm step; no data/model access."""
import json
import os
import socket


def main():
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Requires an explicit allocated Slurm job step")
    info = dict(event="cuda_preflight", node=socket.gethostname(),
                job=os.environ["SLURM_JOB_ID"], step=os.environ["SLURM_STEP_ID"],
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
                slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS"),
                visible_devices=torch.cuda.device_count())
    print(json.dumps(info), flush=True)
    if info["visible_devices"] != 1:
        raise RuntimeError("Worker must see exactly one allocated CUDA device")
    device = torch.cuda.get_device_properties(0)
    value = torch.ones(16, device="cuda").sum().item()
    torch.cuda.synchronize()
    if value != 16.0:
        raise RuntimeError("CUDA allocation check returned unexpected value")
    print(json.dumps(dict(event="cuda_preflight_passed", gpu=device.name,
                          total_memory_bytes=device.total_memory, value=value)), flush=True)
    # Process exits before trainer starts: never retain a competing CUDA context.


if __name__ == "__main__":
    main()
