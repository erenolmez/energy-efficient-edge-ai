param([int[]]$TrackedIds = @(27964, 22072), [switch]$CheckOnly,
      [string]$ExpectedScript = 'run_pc_routes.py',
      [string]$ReportRelativePath = 'pc_tests\model_pipeline\reports\pc_routes_global.json',
      [string]$TrainingMarkerDirectory = '')
$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
$reportPath = Join-Path $studyRoot $ReportRelativePath
$statePath = Join-Path $studyRoot 'pc_tests\model_pipeline\artifacts\pc_routers\thermal_guard.json'
$logPath = Join-Path $studyRoot 'pc_tests\model_pipeline\artifacts\pc_routers\thermal_guard.log'
$mutex = [Threading.Mutex]::new($false, 'Local\Cifar100PcThermalGuard')
if (-not $mutex.WaitOne(0)) { throw 'Thermal guard already running' }
try {
    $identities = @{}
    foreach ($trackedId in $TrackedIds) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $trackedId"
        if (-not $process -or $process.CommandLine -notlike "*$ExpectedScript*" -or $process.CommandLine -notlike '*energy-efficient-edge-ai*') {
            throw "Expected routing process missing: $trackedId"
        }
        $identities[$trackedId] = $process.CreationDate
    }
    $consecutiveHot = 0
    $boundaryCount = $null
    while ($true) {
        $reading = & 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=temperature.gpu --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0 -or $reading -notmatch '^\s*\d+\s*$') {
            throw 'GPU temperature telemetry unavailable; no temperature invented'
        }
        $temperature = [int]$reading.Trim()
        if ($TrainingMarkerDirectory) {
            $completed = @(Get-ChildItem -LiteralPath (Join-Path $studyRoot $TrainingMarkerDirectory) -Filter 'global_*_seed*.json' | Where-Object {
                (Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json).status -eq 'complete'
            }).Count
        } elseif (Test-Path -LiteralPath $reportPath) {
            $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            $completed = @($report.results | Where-Object { $_.router -and $_.status -eq 'complete' }).Count
        } else { $completed = 0 }
        if ($temperature -ge 85) { $consecutiveHot++ } else { $consecutiveHot = 0 }
        if ($consecutiveHot -ge 2 -and $null -eq $boundaryCount) { $boundaryCount = $completed }
        $status = if ($null -eq $boundaryCount) { 'monitoring' } else { 'waiting_for_next_checkpoint' }
        $state = [ordered]@{status=$status; utc=[DateTime]::UtcNow.ToString('o'); gpu_c=$temperature;
            threshold_c=85; sample_seconds=60; consecutive_hot=$consecutiveHot;
            completed_routes=$completed; boundary_start_count=$boundaryCount;
            tracked_process_ids=$TrackedIds; cpu_temperature_available=$false;
            expected_script=$ExpectedScript; training_marker_directory=$TrainingMarkerDirectory}
        $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
        Add-Content -LiteralPath $logPath -Value "$($state.utc) $status gpu=$temperature completed=$completed"
        if ($CheckOnly) { $state | ConvertTo-Json; break }
        $active = @()
        foreach ($trackedId in $TrackedIds) {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $trackedId"
            if ($process) {
                if ($process.CreationDate -ne $identities[$trackedId] -or $process.CommandLine -notlike "*$ExpectedScript*" -or $process.CommandLine -notlike '*energy-efficient-edge-ai*') {
                    throw 'Process identity changed; refusing to stop another process'
                }
                $active += $process
            }
        }
        if ($null -ne $boundaryCount -and $completed -gt $boundaryCount) {
            # The report is atomically replaced by the experiment after each configuration.
            # A just-started unsaved configuration may be interrupted; completed rows survive.
            $state.status = 'thermal_stop_shutdown_requested'
            $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
            foreach ($process in ($active | Sort-Object ProcessId)) {
                Stop-Process -Id $process.ProcessId -ErrorAction Stop
            }
            Add-Content -LiteralPath $logPath -Value 'Completed checkpoint preserved; graceful shutdown requested without /f.'
            & 'C:\Windows\System32\shutdown.exe' /s /t 0 /d p:0:0 /c 'GPU thermal guard: checkpoint preserved; user requested shutdown'
            if ($LASTEXITCODE -ne 0) { throw 'Windows shutdown request failed' }
            break
        }
        if ($active.Count -eq 0) { break }
        Start-Sleep -Seconds 60
    }
} catch {
    Add-Content -LiteralPath $logPath -Value "$([DateTime]::UtcNow.ToString('o')) ERROR $($_.Exception.Message)"
    throw
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
