$ErrorActionPreference = 'Stop'
. 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai\pc_tests\model_pipeline\scripts\require_experiments_resumed.ps1'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$directory = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/b1_engine_profiles_20261028'
$statePath = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/pc_routers/thermal_guard.json'
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.status -ne 'monitoring' -or $null -ne $state.boundary_start_count) { throw 'Thermal latch: never launch' }
$previous = Get-Content -LiteralPath 'pc_tests/model_pipeline/reports/gpu_residency_20261028/audited_summary.json' -Raw | ConvertFrom-Json
$cpu = Get-Content -LiteralPath 'pc_tests/model_pipeline/artifacts/compact_b1_20261028/preparation.json' -Raw | ConvertFrom-Json
if (-not $previous.audit_passed -or $cpu.status -ne 'complete' -or $cpu.results.Count -ne 12) { throw 'Prerequisites incomplete' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'energy-efficient-edge-ai.*(run_|train_|measure_|probe_gpu_residency|profile_b1_engines|prepare_compact_b1)') -or
    ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'watch_pc_thermals.ps1')
})
if ($active.Count) { throw 'Other worker/guard active: refuse overlap' }
if (Test-Path -LiteralPath $directory) { throw 'Existing engine-profile study preserved: no blind restart' }
New-Item -ItemType Directory -Path $directory | Out-Null
$runner = Join-Path $studyRoot 'pc_tests/model_pipeline/experiments/profile_b1_engines.py'
$launcher = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv/Scripts/python.exe') -ArgumentList @('-u',$runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'run.stdout.log') -RedirectStandardError (Join-Path $directory 'run.stderr.log') -PassThru
Start-Sleep -Seconds 2
$identities = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $launcher.Id -or $_.ParentProcessId -eq $launcher.Id) -and $_.CommandLine -like "*$runner*" })
if (-not $identities.Count) { throw 'Worker exited: inspect logs' }
$ids = $identities.ProcessId -join ','
$guardPath = Join-Path $studyRoot 'pc_tests/model_pipeline/scripts/watch_pc_thermals.ps1'
$command = "& '$guardPath' -TrackedIds @($ids) -ExpectedScript 'profile_b1_engines.py' -ReportRelativePath 'pc_tests/model_pipeline/reports/b1_engine_profiles_20261028/study.json'"
$guard = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-Command',$command) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'guard.stdout.log') -RedirectStandardError (Join-Path $directory 'guard.stderr.log') -PassThru
$guardIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($guard.Id)"
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');runner=@($identities|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine);guard_identity=$guardIdentity|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine;scope='validation-only six-engine batch1 profiles'}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $directory 'launch_identity.json') -Encoding utf8
$identities|Select-Object ProcessId,CreationDate,ExecutablePath
Write-Output "Engine-profile thermal guard PID $($guard.Id)"
