$ErrorActionPreference='Stop'
$studyRoot='C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
$out=Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/remaining_work_20261104'
if(Test-Path -LiteralPath $out) { throw 'Campaign artifacts exist; preserve, no restart' }
if(@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }).Count) { throw 'Other Python work active; do not overlap' }
$reading=& 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits
if($LASTEXITCODE -ne 0 -or $reading -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*$') { throw 'Missing GPU telemetry' }
if([int]$Matches[1] -gt 55 -or [int]$Matches[2] -gt 5) { throw 'GPU not cool/idle; no automatic retry' }
New-Item -ItemType Directory -Path $out | Out-Null
$runner=Join-Path $studyRoot 'pc_tests/model_pipeline/experiments/run_remaining_work.py'
$p=Start-Process -FilePath (Join-Path $studyRoot '.cuda-venv/Scripts/python.exe') -ArgumentList @('-u',$runner) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $out 'run.stdout.log') -RedirectStandardError (Join-Path $out 'run.stderr.log') -PassThru
Start-Sleep -Seconds 2
$identities=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.ProcessId -eq $p.Id -or $_.ParentProcessId -eq $p.Id) -and $_.CommandLine -like "*$runner*" })
if(-not $identities.Count) { throw 'Controller exited; preserve logs, no restart' }
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');expected_script='run_remaining_work.py';processes=@($identities|Select-Object ProcessId,ExecutablePath,CommandLine,@{n='creation_unix_ms';e={[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}})}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $out 'launch_identity.json') -Encoding utf8
$identities | Select-Object ProcessId,CreationDate,ExecutablePath
