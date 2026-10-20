# Compatibility entry point; thermal/overlap checks remain in the canonical script.
& (Join-Path $PSScriptRoot 'scripts/start_most_pruned.ps1') @args
