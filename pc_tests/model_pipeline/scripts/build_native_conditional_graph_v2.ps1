$ErrorActionPreference = 'Stop'
$studyRoot='C:\Users\Eren\Desktop\tez\energy-efficient-edge-ai'
$out=Join-Path $studyRoot 'pc_tests/model_pipeline/artifacts/native_conditional_graph_v2_20261104'
if (Test-Path -LiteralPath $out) { throw 'Existing graph probe preserved; no blind rebuild/restart' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }).Count) { throw 'Python work active; do not overlap' }
New-Item -ItemType Directory -Path $out | Out-Null
$nvcc='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe'
$compiler='C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64'
Set-Location -LiteralPath $out
& $nvcc '-shared' '-rdc=true' '-arch=sm_75' '--fmad=false' '-allow-unsupported-compiler' '-ccbin' $compiler '-o' (Join-Path $out 'native_conditional_graph.dll') (Join-Path $studyRoot 'pc_tests/model_pipeline/runtime/native_conditional_graph_v2.cu') 2>&1 | Tee-Object -FilePath (Join-Path $out 'build.log')
if ($LASTEXITCODE -ne 0) { throw 'Native graph build failed; inspect log, no auto retry' }
