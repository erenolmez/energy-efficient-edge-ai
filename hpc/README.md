# HPC preparation (local files only)

No remote login, upload, package installation, allocation or job submission has
been performed by this preparation. Do not run the Windows PC experiment
launchers on Legion: their paths, process checks and thermal controls are
machine-specific. A Linux training adapter needs explicit implementation first.

## Intended division of work

- PC RTX 2060: deployment latency/energy, batch-size comparisons, and same-device
  router training-energy payback.
- HPC: larger classifier training/fine-tuning, clean-data router supervision,
  independent seeds and training sweeps. Queue time and tiny jobs can negate any
  acceleration; a single roughly minute-long router fit is not the main target.
- HPC inference is a separate hardware study, not a replacement for PC results.
  TensorRT engines must be rebuilt/verified for the target GPU/software stack.
  Multiple GPUs only help with explicitly parallel work; allocation alone does
  not accelerate the existing single-GPU implementation.

## Read-only access checks to run after connecting

Use your own username with `hpc-legionlogin.polito.it`; VPN may be required.
Keep credentials/private keys out of the repository and chat. Confirm VS Code
Server/SSH forwarding policy with the service if using Remote-SSH.

```bash
hostname
sinfo -o '%P %a %l %G %D'
squeue -u "$USER"
module avail
printf 'scratch: %s\n' "${SCRATCH_FLASH:-UNSET}"
```

These inspect scheduler/environment state only. GPU probing, compilation,
environment setup and compute belong in an allocated compute-node session.
Check the current login banner and node states: the earlier electrical-issue
warning must not be assumed resolved. Do not request specific unhealthy nodes.

## First allocated GPU check (not submitted yet)

After identifying your permitted partition and a Linux Python environment with
PyTorch, from the project directory:

```bash
sbatch --partition=YOUR_CONFIRMED_GPU_PARTITION \
  --export=ALL,HPC_PYTHON=/absolute/path/to/your/linux/python \
  hpc/preflight.sbatch
```

The supplied job requests one GPU for at most ten minutes and runs diagnostics
only, not training. It uses the site-provided SCRATCH_FLASH variable, creates a
small unique job directory there, and never replaces previous outputs. Review
the job output before any training port or bulk artifact transfer. Do not copy
the Windows `.cuda-venv` to Linux.

## Storage and reproducibility

The supplied guide says use BeeGFS `SCRATCH_FLASH` for active job data, not home
for simulation I/O. Scratch files older than 30 days can be deleted, and scratch
has no data-protection guarantee. Copy important code/results back to persistent
storage and maintain an independent backup. Transfer only an explicit manifest
of required inputs and hashes, not the entire artifact tree.

Record GPU model, allocation, driver/CUDA/PyTorch, seed, split hashes, exact code
revision, training recipe, checkpoint hashes and energy scope. GPU-only HPC
energy is not whole-node energy. Shared-device activity contaminates counters;
an attributable energy study needs suitable isolation and telemetry permissions.

Source: user-provided `C:/Users/Eren/Desktop/ders/Thesis/hpcpolito_guide.pdf`,
Basic Rules and chapters 3, 4, 6, 7. Partition availability/policies must be
confirmed on the live cluster rather than inferred from the guide.
