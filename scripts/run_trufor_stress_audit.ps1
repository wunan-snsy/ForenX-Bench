param(
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "external\TruFor\.venv\Scripts\python.exe"
$TruForSrc = Join-Path $Repo "external\TruFor\test_docker\src"
$Inputs = Join-Path $Repo "results\trufor_stress_inputs"
$Outputs = Join-Path $Repo "results\trufor_stress_outputs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "TruFor virtual-environment Python was not found: $Python"
}

& python (Join-Path $PSScriptRoot "prepare_trufor_stress_audit.py")
Push-Location $TruForSrc
try {
    & $Python "trufor_test.py" -gpu $Gpu -in (Join-Path $Inputs "*.jpg") -out $Outputs
    if ($LASTEXITCODE -ne 0) { throw "TruFor JPEG inference failed with exit code $LASTEXITCODE" }
    & $Python "trufor_test.py" -gpu $Gpu -in (Join-Path $Inputs "*.png") -out $Outputs
    if ($LASTEXITCODE -ne 0) { throw "TruFor PNG inference failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
& python (Join-Path $PSScriptRoot "evaluate_trufor_stress_audit.py")
if ($LASTEXITCODE -ne 0) { throw "Stress evaluation failed with exit code $LASTEXITCODE" }
