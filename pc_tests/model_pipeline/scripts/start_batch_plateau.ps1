$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$out = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/batch_plateau_20261103'
if (Test-Path -LiteralPath $out) { throw 'Existing diagnostic preserved; never duplicate or blindly restart' }
$others = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -or
    ($_.Name -match 'powershell' -and $_.CommandLine -match 'watch_pc_thermals.ps1')
})
if ($others.Count) { throw 'Python worker or legacy guard active; inspect before GPU timing' }
$reading = & 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or $reading -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*$') { throw 'GPU preflight telemetry unavailable' }
if ([int]$Matches[1] -ge 80 -or [int]$Matches[2] -gt 5) { throw 'GPU not cool/idle; defer without starting' }
New-Item -ItemType Directory -Path $out | Out-Null
$runner = Join-Path $studyRoot 'pc_tests/model_pipeline/experiments/diagnose_batch_plateau.py'
$process = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv/Scripts/python.exe') -ArgumentList @('-u', $runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $out 'run.stdout.log') -RedirectStandardError (Join-Path $out 'run.stderr.log') -PassThru
Start-Sleep -Seconds 2
$identities = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $process.Id -or $_.ParentProcessId -eq $process.Id) -and $_.CommandLine -like "*$runner*"
})
if (-not $identities.Count) { throw 'Worker exited during startup; preserve logs, no automatic restart' }
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');expected_script='diagnose_batch_plateau.py';runner=@($identities | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine);guard='In-process NVML sampling every1s, conservative85C latch at next saved measurement, no shutdown';expected_windows=27} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'launch_identity.json') -Encoding utf8
$identities | Select-Object ProcessId,CreationDate,ExecutablePath
