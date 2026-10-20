param([int[]]$TrackedIds = @(27964, 22072), [switch]$CheckOnly,
      [string]$ExpectedScript = 'run_pc_routes.py',
      [string]$ReportRelativePath = 'pc_tests\model_pipeline\reports\pc_routes_global.json',
      [string]$TrainingMarkerDirectory = '')
# Compatibility entry point; keep named parameter forwarding explicit.
& (Join-Path $PSScriptRoot 'scripts/watch_pc_thermals.ps1') @PSBoundParameters @args
