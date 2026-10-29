$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
. "$studyRoot/pc_tests/model_pipeline/scripts/require_experiments_resumed.ps1"
$state = Get-Content pc_tests/model_pipeline/artifacts/pc_routers/thermal_guard.json -Raw | ConvertFrom-Json
if ($state.status -ne 'monitoring' -or $null -ne $state.boundary_start_count) { throw 'Thermal latch' }
$previous = Get-Content pc_tests/model_pipeline/reports/batch8_grouping_20261029/audited_summary.json -Raw | ConvertFrom-Json
if (-not $previous.audit_passed) { throw 'Resident screen audit incomplete' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and (($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'energy-efficient-edge-ai.*(compare_|screen_|confirm_|pilot_|run_|train_|measure_|probe_|profile_|diagnose_|refresh_|prepare_compact)') -or
    ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'watch_pc_thermals.ps1'))
})
if ($active.Count) { throw 'Worker or guard already active' }
$directory = "$studyRoot/pc_tests/model_pipeline/artifacts/dispatch_controls_20261029"
if (Test-Path -LiteralPath $directory) { throw 'Existing compact screen preserved' }
New-Item -ItemType Directory -Path $directory | Out-Null
$runner = "$studyRoot/pc_tests/model_pipeline/experiments/profile_dispatch_controls.py"
$launcher = Start-Process -FilePath "$studyRoot/.cuda-venv/Scripts/python.exe" -ArgumentList @('-u',$runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput "$directory/run.stdout.log" -RedirectStandardError "$directory/run.stderr.log" -PassThru
Start-Sleep -Seconds 2
$identities = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $launcher.Id -or $_.ParentProcessId -eq $launcher.Id) -and $_.CommandLine -like "*$runner*" })
if (-not $identities.Count) { throw 'Worker exited; inspect logs' }
$ids = $identities.ProcessId -join ','
$guardPath = "$studyRoot/pc_tests/model_pipeline/scripts/watch_pc_thermals.ps1"
$command = "& '$guardPath' -TrackedIds @($ids) -ExpectedScript 'profile_dispatch_controls.py' -ReportRelativePath 'pc_tests/model_pipeline/reports/dispatch_controls_20261029/study.json'"
$guard = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-Command',$command) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput "$directory/guard.stdout.log" -RedirectStandardError "$directory/guard.stderr.log" -PassThru
$guardIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($guard.Id)"
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');runner=@($identities|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine);guard_identity=$guardIdentity|Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine;scope='Validation-onlypairedsameclassifier,FP32TRTselection,aggregatecoldenergyandsensoraudit,no testdata'}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath "$directory/launch_identity.json" -Encoding utf8
$identities|Select-Object ProcessId,CreationDate,ExecutablePath
Write-Output "Compact screen thermal guard PID $($guard.Id)"
