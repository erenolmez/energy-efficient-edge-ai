# Compatibility entry point; thermal/overlap checks remain in the canonical script.
& (Join-Path $PSScriptRoot 'scripts/start_p80_batches.ps1') @args
