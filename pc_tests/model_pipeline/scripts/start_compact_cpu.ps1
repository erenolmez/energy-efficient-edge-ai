$ErrorActionPreference = 'Stop'
. 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai\pc_tests\model_pipeline\scripts\require_experiments_resumed.ps1'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$logs = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/compact_b1_launch_20261028'
if (Test-Path -LiteralPath $logs) { throw 'CPU launch already exists: inspect, never blindly restart' }
& '.\.cuda-venv\Scripts\python.exe' -c 'from pc_tests.model_pipeline.experiments.prepare_compact_b1 import require_predecessor_done; require_predecessor_done()'
if ($LASTEXITCODE -ne 0) { throw 'CPU predecessor/idle check refused launch' }
New-Item -ItemType Directory -Path $logs | Out-Null
$runner = Join-Path $studyRoot 'pc_tests/model_pipeline/experiments/prepare_compact_b1.py'
$launcher = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv/Scripts/python.exe') -ArgumentList @('-u',$runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'run.stdout.log') -RedirectStandardError (Join-Path $logs 'run.stderr.log') -PassThru
Start-Sleep -Seconds 2
$identities = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $launcher.Id -or $_.ParentProcessId -eq $launcher.Id) -and $_.CommandLine -like "*$runner*" })
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');scope='CPU only, no GPU thermal guard claimed';runner=@($identities|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine);expected_fits=12}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $logs 'launch_identity.json') -Encoding utf8
$identities|Select-Object ProcessId,CreationDate,ExecutablePath
