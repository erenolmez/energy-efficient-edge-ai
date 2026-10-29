$ErrorActionPreference = 'Stop'
. 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai\pc_tests\model_pipeline\scripts\require_experiments_resumed.ps1'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$directory = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/gpu_residency_20261028'
$statePath = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/pc_routers/thermal_guard.json'
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.status -ne 'monitoring' -or $null -ne $state.boundary_start_count) { throw 'Thermal latch: never launch' }
$receipt = Get-Content -LiteralPath 'pc_tests/model_pipeline/reports/correctness_b1_20261028/stop_receipt.json' -Raw | ConvertFrom-Json
$oldReport = 'pc_tests/model_pipeline/reports/correctness_b1_20261028/study.json'
if ($receipt.status -ne 'stopped_by_user' -or (Get-FileHash -LiteralPath $oldReport -Algorithm SHA256).Hash.ToLower() -ne $receipt.study_sha256) { throw 'Stopped baseline identity changed' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'energy-efficient-edge-ai.*(run_|train_|measure_|probe_gpu_residency)') -or
    ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'watch_pc_thermals.ps1')
})
if ($active.Count) { throw 'Worker/guard still active: refuse overlap' }
if (Test-Path -LiteralPath $directory) { throw 'Existing probe preserved; no blind restart' }
New-Item -ItemType Directory -Path $directory | Out-Null
$runner = Join-Path $studyRoot 'pc_tests/model_pipeline/experiments/probe_gpu_residency.py'
$launcher = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv/Scripts/python.exe') -ArgumentList @('-u',$runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'run.stdout.log') -RedirectStandardError (Join-Path $directory 'run.stderr.log') -PassThru
Start-Sleep -Seconds 2
$identities = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $launcher.Id -or $_.ParentProcessId -eq $launcher.Id) -and $_.CommandLine -like "*$runner*" })
if (-not $identities.Count) { throw 'Worker exited before guard: inspect logs' }
$ids = $identities.ProcessId -join ','
$guardPath = Join-Path $studyRoot 'pc_tests/model_pipeline/scripts/watch_pc_thermals.ps1'
$command = "& '$guardPath' -TrackedIds @($ids) -ExpectedScript 'probe_gpu_residency.py' -ReportRelativePath 'pc_tests/model_pipeline/reports/gpu_residency_20261028/study.json'"
$guard = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-Command',$command) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'guard.stdout.log') -RedirectStandardError (Join-Path $directory 'guard.stderr.log') -PassThru
$guardIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($guard.Id)"
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');runner=@($identities|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine);guard_identity=$guardIdentity|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine;orders=3;reserve_mib=1024}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $directory 'launch_identity.json') -Encoding utf8
$identities|Select-Object ProcessId,CreationDate,ExecutablePath
Write-Output "Residency thermal guard PID $($guard.Id)"
