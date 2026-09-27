param([ValidateSet("trufor", "mvss", "both")][string]$Model = "both", [int]$Gpu = 0)
$ErrorActionPreference = "Stop"; $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
& python (Join-Path $PSScriptRoot "prepare_coverage_inputs.py")
if ($LASTEXITCODE) { throw "COVERAGE input preparation failed" }
$Images = Join-Path $Repo "results\coverage_inputs"
if ($Model -in @("trufor", "both")) {
  $Python = Join-Path $Repo "external\TruFor\.venv\Scripts\python.exe"; $Src = Join-Path $Repo "external\TruFor\test_docker\src"
  Push-Location $Src; try { foreach ($Class in @("authentic", "tampered")) { $glob = Join-Path (Join-Path $Images $Class) "*.tif"; $out=Join-Path $Repo "results\trufor_coverage_$Class"; & $Python trufor_test.py -gpu $Gpu -in $glob -out $out; if ($LASTEXITCODE) { throw "TruFor failed: $Class" } } } finally { Pop-Location }
}
if ($Model -in @("mvss", "both")) { foreach ($Class in @("authentic", "tampered")) { $glob = Join-Path (Join-Path $Images $Class) "*.tif"; $out=Join-Path $Repo "results\mvss_coverage_$Class"; & python (Join-Path $PSScriptRoot run_mvss_inference.py) --input $glob --output $out --gpu $Gpu; if ($LASTEXITCODE) { throw "MVSS failed: $Class" } } }
if ($Model -eq "both") { & python (Join-Path $PSScriptRoot evaluate_coverage_models.py); if ($LASTEXITCODE) { throw "COVERAGE evaluation failed" } }
