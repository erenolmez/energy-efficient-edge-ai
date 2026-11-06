$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
Set-Location -LiteralPath $studyRoot
$out = Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/router_warmup_20261103'
if (Test-Path -LiteralPath $out) { throw 'Existing warm-up study preserved; no duplicate launch or blind restart' }
$others = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -or
    ($_.Name -match 'powershell' -and $_.CommandLine -match 'watch_pc_thermals.ps1')
})
if ($others.Count) { throw 'Python process/legacy guard active; no overlapping GPU fits' }
$reading = & 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or $reading -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*$') { throw 'GPU telemetry unavailable' }
if ([int]$Matches[1] -gt 70 -or [int]$Matches[2] -gt 5) { throw 'GPU not cool/idle; study not launched' }
New-Item -ItemType Directory -Path $out | Out-Null
$runner = Join-Path $studyRoot 'pc_tests/model_pipeline/experiments/compare_router_warmup.py'
$process = Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv/Scripts/python.exe') -ArgumentList @('-u',$runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $out 'controller.stdout.log') -RedirectStandardError (Join-Path $out 'controller.stderr.log') -PassThru
Start-Sleep -Seconds 2
$identities = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $process.Id -or $_.ParentProcessId -eq $process.Id) -and $_.CommandLine -like "*$runner*"
})
if (-not $identities.Count) { throw 'Controller exited; preserve logs and inspect, never automatically restart' }
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');expected_script='compare_router_warmup.py';controller=@($identities | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine);expected_fits=36;expected_epochs=720;child_identity_files='b*_*.identity.json';guard='Per-child one-second NVML;70C/no thermal-flag start gate;85C latch after saved epoch; no shutdown'} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'launch_identity.json') -Encoding utf8
$identities | Select-Object ProcessId,CreationDate,ExecutablePath
