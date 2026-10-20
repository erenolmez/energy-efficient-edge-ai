$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$statePath = Join-Path $studyRoot 'pc_tests\model_pipeline\artifacts\pc_routers\thermal_guard.json'
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.status -ne 'monitoring') { throw 'Thermal state latched or not monitoring; never launch' }
$previous = Get-Content -LiteralPath 'pc_tests\model_pipeline\reports\random_b64\random_routes_b64.json' -Raw | ConvertFrom-Json
if ($previous.status -ne 'complete' -or $previous.results.Count -ne 495) { throw 'Random batch64 study incomplete; wait' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(run_random_routes(_b64)?|run_most_pruned(_batches)?|run_p80_batches|run_pc_routes|run_correctness_routes|gpu_static_sweep|train_pc_batch_routers|train_correctness_router)\.py'
})
if ($active.Count) { throw 'Another GPU study is active; do not duplicate or overlap' }
$guards = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*watch_pc_thermals.ps1*'
})
if ($guards.Count) { throw 'Previous thermal guard has not exited; retry next scheduled check' }
& '.\.cuda-venv\Scripts\python.exe' 'pc_tests\model_pipeline\summarize_random_routes_b64.py'
if ($LASTEXITCODE -ne 0) { throw 'Random64 summary failed; inspect' }
$summary = Get-Content -LiteralPath 'pc_tests\model_pipeline\reports\random_b64\random_comparison_summary.json' -Raw | ConvertFrom-Json
if (-not $summary.complete -or $summary.rows.Count -ne 72) { throw 'Random64 summary incomplete' }
& '.\.cuda-venv\Scripts\python.exe' 'pc_tests\model_pipeline\run_p80_batches.py' '--prepare-only'
if ($LASTEXITCODE -ne 0) { throw 'p80 preparation failed; preserve existing reports' }
$directory = Join-Path $studyRoot 'pc_tests\model_pipeline\artifacts\p80_b8_32_64_128'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$runnerPath = Join-Path $studyRoot 'pc_tests\model_pipeline\run_p80_batches.py'
$worker = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv\Scripts\python.exe') -ArgumentList @('-u',$runnerPath) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'run.stdout.log') -RedirectStandardError (Join-Path $directory 'run.stderr.log') -PassThru
Start-Sleep -Seconds 3
$identities = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like "*$runnerPath*" })
if (-not $identities.Count) { throw 'p80 worker exited before guard startup; inspect logs' }
$idText = $identities.ProcessId -join ','
$guardPath = Join-Path $studyRoot 'pc_tests\model_pipeline\watch_pc_thermals.ps1'
$guardCommand = "& '$guardPath' -TrackedIds @($idText) -ExpectedScript 'run_p80_batches.py' -ReportRelativePath 'pc_tests\model_pipeline\reports\p80_b8_32_64_128\p80_routes.json'"
$watcher = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-Command',$guardCommand) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'guard.stdout.log') -RedirectStandardError (Join-Path $directory 'guard.stderr.log') -PassThru
$guardIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($watcher.Id)"
[ordered]@{utc=[DateTime]::UtcNow.ToString('o'); runner=@($identities | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine); guard_pid=$watcher.Id; guard_identity=$guardIdentity | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine; expected_sweeps=440} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $directory 'launch_identity.json') -Encoding utf8
$identities | Select-Object ProcessId,CreationDate,ExecutablePath
Write-Output "p80 thermal guard PID $($watcher.Id)"
