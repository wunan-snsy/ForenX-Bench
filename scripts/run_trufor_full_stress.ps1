param([int]$Gpu = 0)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "external\TruFor\.venv\Scripts\python.exe"
$TruForSrc = Join-Path $Repo "external\TruFor\test_docker\src"
$Inputs = Join-Path $Repo "results\trufor_full_stress_inputs"
$Outputs = Join-Path $Repo "results\trufor_full_stress_outputs"
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing TruFor Python: $Python" }
& python (Join-Path $PSScriptRoot "prepare_trufor_full_stress.py")
Push-Location $TruForSrc
try {
    foreach ($condition in @("jpeg_q80", "jpeg_q40", "resize_0.5", "resize_0.25")) {
        $extension = if ($condition.StartsWith("jpeg")) { "*.jpg" } else { "*.png" }
        foreach ($class in @("authentic", "spliced")) {
            $inputGlob = Join-Path (Join-Path (Join-Path $Inputs $condition) $class) $extension
            $outputDir = Join-Path (Join-Path $Outputs $condition) $class
            & $Python "trufor_test.py" -gpu $Gpu -in $inputGlob -out $outputDir
            if ($LASTEXITCODE -ne 0) { throw "Inference failed: $condition/$class" }
        }
    }
}
finally { Pop-Location }
& python (Join-Path $PSScriptRoot "evaluate_trufor_full_stress.py")
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
