# Lifecycle interlock shared by new experimental launchers. No automatic clearing.
$pauseFile = 'C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai\pc_tests\model_pipeline\reports\experiment_pause.json'
if (Test-Path -LiteralPath $pauseFile) {
    $pauseState = Get-Content -LiteralPath $pauseFile -Raw | ConvertFrom-Json
    if ($pauseState.status -eq 'paused_by_user') {
        throw 'Experiments paused by user. Wait for explicit human resume; do not auto-clear this latch.'
    }
}
