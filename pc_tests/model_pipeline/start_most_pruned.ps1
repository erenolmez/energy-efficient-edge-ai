$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$state = Get-Content -LiteralPath 'pc_tests\model_pipeline\artifacts\pc_routers\thermal_guard.json' -Raw | ConvertFrom-Json
if ($state.status -ne 'monitoring') { throw 'Thermal state latched; never launch' }
$random = Get-Content -LiteralPath 'pc_tests\model_pipeline\reports\random_b128\random_routes_b128.json' -Raw | ConvertFrom-Json
if ($random.status -ne 'complete' -or $random.results.Count -ne 495) { throw 'Random comparison incomplete; wait' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(run_random_routes|run_most_pruned|run_pc_routes|run_correctness_routes|gpu_static_sweep|train_pc_batch_routers|train_correctness_router)\.py'
})
if ($active.Count) { throw 'GPU experiment still active; do not duplicate' }
$guards = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*watch_pc_thermals.ps1*'
})
if ($guards.Count) { throw 'Previous thermal guard still exiting; retry next check, do not replace blindly' }
& '.\.cuda-venv\Scripts\python.exe' 'pc_tests\model_pipeline\summarize_random_routes.py'
if ($LASTEXITCODE -ne 0) { throw 'Random summary failed; inspect before starting next stage' }
$summary = Get-Content -LiteralPath 'pc_tests\model_pipeline\reports\random_b128\random_comparison_summary.json' -Raw | ConvertFrom-Json
if (-not $summary.complete -or $summary.rows.Count -ne 72) { throw 'Random comparison audit incomplete' }
$directory = Join-Path $studyRoot 'pc_tests\model_pipeline\artifacts\most_pruned_b128'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$runnerPath = Join-Path $studyRoot 'pc_tests\model_pipeline\run_most_pruned.py'
$worker = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv\Scripts\python.exe') -ArgumentList @('-u', $runnerPath) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'run.stdout.log') -RedirectStandardError (Join-Path $directory 'run.stderr.log') -PassThru
Start-Sleep -Seconds 3
$identities = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like "*$runnerPath*" })
if (-not $identities.Count) { throw 'Runner exited before guard startup; inspect logs' }
$idText = $identities.ProcessId -join ','
$guardPath = Join-Path $studyRoot 'pc_tests\model_pipeline\watch_pc_thermals.ps1'
$guardCommand = "& '$guardPath' -TrackedIds @($idText) -ExpectedScript 'run_most_pruned.py' -ReportRelativePath 'pc_tests\model_pipeline\reports\most_pruned_b128\most_pruned_routes.json'"
$watcher = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-Command',$guardCommand) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory 'guard.stdout.log') -RedirectStandardError (Join-Path $directory 'guard.stderr.log') -PassThru
[ordered]@{utc=[DateTime]::UtcNow.ToString('o'); runner=@($identities | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine); guard_pid=$watcher.Id; expected_sweeps=110} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $directory 'launch_identity.json') -Encoding utf8
$identities | Select-Object ProcessId,CreationDate,ExecutablePath
Write-Output "Most-pruned guard PID $($watcher.Id)"
