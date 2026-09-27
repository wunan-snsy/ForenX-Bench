param([ValidateSet("trufor", "mvss", "both")][string]$Model = "both", [int]$Gpu = 0)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Inputs = Join-Path $Repo "results\casia_v1_full_stress_inputs"
& python (Join-Path $PSScriptRoot "prepare_casia_v1_full_stress.py")
if ($LASTEXITCODE -ne 0) { throw "CASIA stress-input preparation failed" }
if ($Model -in @("trufor", "both")) {
    $Python = Join-Path $Repo "external\TruFor\.venv\Scripts\python.exe"
    $Source = Join-Path $Repo "external\TruFor\test_docker\src"
    Push-Location $Source
    try {
        foreach ($Condition in @("jpeg_q80", "jpeg_q40", "resize_0.5", "resize_0.25")) {
            foreach ($Class in @("authentic", "tampered")) {
                $InputGlob = Join-Path (Join-Path (Join-Path $Inputs $Condition) $Class) "*.jpg"
                $Output = Join-Path (Join-Path (Join-Path $Repo "results\trufor_casia_v1_full_stress_outputs") $Condition) $Class
                & $Python "trufor_test.py" -gpu $Gpu -in $InputGlob -out $Output
                if ($LASTEXITCODE -ne 0) { throw "TruFor failed: $Condition/$Class" }
            }
        }
    } finally { Pop-Location }
}
if ($Model -in @("mvss", "both")) {
    foreach ($Condition in @("jpeg_q80", "jpeg_q40", "resize_0.5", "resize_0.25")) {
        foreach ($Class in @("authentic", "tampered")) {
            $InputGlob = Join-Path (Join-Path (Join-Path $Inputs $Condition) $Class) "*.jpg"
            $Output = Join-Path (Join-Path (Join-Path $Repo "results\mvss_casia_v1_full_stress_outputs") $Condition) $Class
            & python (Join-Path $PSScriptRoot "run_mvss_inference.py") --input $InputGlob --output $Output --gpu $Gpu
            if ($LASTEXITCODE -ne 0) { throw "MVSS-Net failed: $Condition/$Class" }
        }
    }
}
if ($Model -eq "both") {
    & python (Join-Path $PSScriptRoot "evaluate_casia_v1_full_stress.py")
    if ($LASTEXITCODE -ne 0) { throw "CASIA stress evaluation failed" }
}
