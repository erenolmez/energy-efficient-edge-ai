#!/bin/bash
#SBATCH --job-name=router-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=hpc-preflight-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Must run through SLURM, never on the login node}"
: "${SCRATCH_FLASH:?Confirm the site scratch mount first}"
: "${HPC_PYTHON:?Supply the absolute path to a Linux PyTorch Python}"
test -x "$HPC_PYTHON"
task_dir=$(mktemp -d "${SCRATCH_FLASH}/router-preflight-${SLURM_JOB_ID}-XXXXXX")
cd "$task_dir"
hostname
date -u
printf 'job=%s scratch=%s\n' "$SLURM_JOB_ID" "$task_dir"
nvidia-smi
module list 2>&1 || true
srun --ntasks=1 "$HPC_PYTHON" -c 'import sys,torch; print("python",sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print("gpu",torch.cuda.get_device_name(0)); x=torch.ones(16,device="cuda"); print("smoke_sum",x.sum().item()); torch.cuda.synchronize()'
