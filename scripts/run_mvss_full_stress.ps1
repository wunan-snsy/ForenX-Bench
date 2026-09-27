param([int]$Gpu = 0)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $PSScriptRoot "run_mvss_inference.py"
$Inputs = Join-Path $Repo "results\trufor_full_stress_inputs"
$Outputs = Join-Path $Repo "results\mvss_full_stress_outputs"
foreach ($condition in @("jpeg_q80", "jpeg_q40", "resize_0.5", "resize_0.25")) {
    $extension = if ($condition.StartsWith("jpeg")) { "*.jpg" } else { "*.png" }
    foreach ($class in @("authentic", "spliced")) {
        $inputGlob = Join-Path (Join-Path (Join-Path $Inputs $condition) $class) $extension
        $outputDir = Join-Path (Join-Path $Outputs $condition) $class
        & python $Runner --input $inputGlob --output $outputDir --gpu $Gpu
        if ($LASTEXITCODE -ne 0) { throw "MVSS inference failed: $condition/$class" }
    }
}
& python (Join-Path $PSScriptRoot "evaluate_mvss_full_stress.py")
if ($LASTEXITCODE -ne 0) { throw "MVSS stress evaluation failed" }
