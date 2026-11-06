# Run the Adam fine-tuning tail on Slurm

The original PC worker keeps running unchanged. It may duplicate some tail
cases if it catches up. Do not launch a second PC worker. The importer indexes
HPC results for the assigned tail and keeps all original PC files intact.

From PowerShell, upload the prepared bundle yourself:

```powershell
scp 'C:\Users\Eren\.codex\finetune-handoff\hpc-input.tar.gz' 'eolmez@hpc-legionlogin.polito.it:~/adam-finetune-tail-20261105.tar.gz'
```

On the HPC login node, paste this block. It prompts for the CUDA-capable
Python executable and shared directory containing `cifar-100-python/train`.
It checks the package versions before submitting three one-GPU array tasks.
No dependencies or datasets are downloaded.

```bash
(
  set -euo pipefail
  read -r -p 'Absolute CUDA-environment Python path: ' PY
  read -r -p 'Absolute directory containing cifar-100-python/: ' DATA
  test -x "$PY"
  test -r "$DATA/cifar-100-python/train"
  ROOT="$HOME/adam-finetune-tail-20261105"
  mkdir "$ROOT"
  tar -xzf "$HOME/adam-finetune-tail-20261105.tar.gz" -C "$ROOT"
  "$PY" -B -c 'import sys; sys.path.insert(0, sys.argv[1]); import split_finetune as s; from pathlib import Path; p=s.read(Path(sys.argv[1])/"handoff.json"); s.require(s.versions()==p["versions"], "Expected Python packages: "+str(p["versions"]))' "$ROOT"
  sbatch --array=0-2 --output="$ROOT/slurm-%A_%a.log" "$ROOT/split_finetune.sbatch" "$ROOT" "$PY" "$DATA"
  printf 'Collect from: %s/results/export\n' "$ROOT"
)
```

Each completed case automatically publishes its best checkpoint, report, and
SHA-256 receipt. Failed cases publish no result. Do not blindly resubmit into
an existing attempt directory. If setup fails before submission, keep the
extracted files and fix the reported issue before submitting.

After the HPC tail and PC prefix finish, paste this one PowerShell block:

```powershell
Set-Location 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
$handoff = 'C:\Users\Eren\.codex\finetune-handoff'
$remote = Read-Host 'Paste the absolute Collect from path printed on HPC'
& .\.cuda-venv\Scripts\python.exe -B .\pc_tests\model_pipeline\scripts\split_finetune.py collect "$handoff\handoff.json" "eolmez@hpc-legionlogin.polito.it:$remote" "$handoff\imported"
```

The importer verifies the completed cases and checkpoints and writes
`imported/model_index.csv` alongside independent copies of all 63 best
checkpoints. It does not load model files or move or overwrite existing PC results.
