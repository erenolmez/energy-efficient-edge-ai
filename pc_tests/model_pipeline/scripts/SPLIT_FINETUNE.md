# Split the PC Adam campaign across PC and Slurm

For the live, no-stop tail handoff, use [the tail commands](SPLIT_FINETUNE_TAIL.md).
The instructions below are only for the distinct stop-and-replace mode.

This is separate from the CIFAR32 sweep. Do not run the original all-case PC
worker alongside the split workers. It does not read case ownership dynamically.
Stop it only after agreeing how to handle its in-progress case. Its existing
reports and models must stay in place; no interrupted checkpoint is resumed.

From the repository root in PowerShell, once the original worker is stopped:

```powershell
$py = '.\.cuda-venv\Scripts\python.exe'
$split = '.\pc_tests\model_pipeline\scripts\split_finetune.py'
$handoff = 'C:\Users\Eren\.codex\finetune-handoff'
& $py -B $split prepare $handoff --workers 3
if ($LASTEXITCODE -ne 0) { throw 'Handoff preparation failed' }
```

The preparation checks the original plan and checkpoint hashes, assigns each
remaining case once, and creates `finetune-handoff/hpc-input.tar.gz`. This
bundle contains only runner code, the split manifest, and the required original
baseline checkpoints and metadata. It contains no dataset or credentials.

Start the assigned PC shard from the repository root:

```powershell
& $py -B -u $split worker --root . --handoff "$handoff\handoff.json" --owner pc --data 'C:\Users\Eren\Desktop\tez\pruning + quantization\data' --out "$handoff\pc-results"
```

In a separate PowerShell window, upload the bundle to the login host yourself:

```powershell
scp 'C:\Users\Eren\.codex\finetune-handoff\hpc-input.tar.gz' 'eolmez@hpc-legionlogin.polito.it:~/hpc-input.tar.gz'
```

On the HPC login node, replace `PY` with the CUDA-capable environment's
Python executable and `DATA` with the shared directory containing the
extracted CIFAR-100 Python dataset. The runner requires the same Torch,
TorchVision, Torch-Pruning, NumPy and Pillow package versions as the PC; they
are listed in `handoff.json`. No automatic installation or dataset download
is performed. This example requests three independent one-GPU array tasks.
Adjust the range to match the handoff if you prepared fewer HPC workers.

```bash
ROOT="$HOME/adam-finetune-split"
PY=/absolute/path/to/venv/bin/python
DATA=/absolute/path/to/cifar100-parent
mkdir "$ROOT" && tar -xzf "$HOME/hpc-input.tar.gz" -C "$ROOT" &&
  sbatch --array=0-2 --output="$ROOT/slurm-%A_%a.log" "$ROOT/split_finetune.sbatch" "$ROOT" "$PY" "$DATA"
printf 'Collect from: %s/results/export\n' "$ROOT"
```

Each successful case publishes its best checkpoint, `study.json` and a SHA-256
receipt in a single directory. A failed case publishes no completed result;
do not blindly resubmit a shard into the same attempt directory.

After both PC and HPC finish, run one command from the repository root. Replace
the remote path below with the `Collect from` path printed above:

```powershell
$handoff = 'C:\Users\Eren\.codex\finetune-handoff'
& .\.cuda-venv\Scripts\python.exe -B .\pc_tests\model_pipeline\scripts\split_finetune.py collect "$handoff\handoff.json" 'eolmez@hpc-legionlogin.polito.it:/absolute/path/to/results/export' "$handoff\imported"
```

The importer stages the transfer, checks case ownership, reports, split,
epoch records, and best-checkpoint hashes, and refuses incomplete or conflicting
results. It creates `imported/model_index.csv` with local paths for all 63 best
models without rewriting original PC outputs. It does not unpickle checkpoints.
