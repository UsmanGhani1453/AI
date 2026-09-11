# convert_to_gguf.ps1
#
# Converts the merged Hugging Face model (produced by
# training/merge_adapter.py) into an fp16 GGUF file, using llama.cpp's
# conversion script. No C++ build is required for this step.
#
# Run from the project root:
#   .\deployment\convert_to_gguf.ps1
#
# Requires: git, Python (the project's venv is fine).

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MergedModel = Join-Path $ProjectRoot "models\merged"
$LlamaCppDir = Join-Path $ProjectRoot "deployment\llama.cpp"
$GgufOutDir  = Join-Path $ProjectRoot "models\gguf"
$GgufOutFile = Join-Path $GgufOutDir "truck-dispatch-f16.gguf"

if (-not (Test-Path $MergedModel)) {
    Write-Error "Merged model not found at $MergedModel. Run 'python training\merge_adapter.py' first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $GgufOutDir | Out-Null

if (-not (Test-Path $LlamaCppDir)) {
    Write-Host "Cloning llama.cpp (conversion script only, no build needed)..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp $LlamaCppDir
}

Write-Host "Installing conversion script dependencies..."
pip install -r (Join-Path $LlamaCppDir "requirements\requirements-convert_hf_to_gguf.txt")

Write-Host "Converting merged model to fp16 GGUF..."
python (Join-Path $LlamaCppDir "convert_hf_to_gguf.py") `
    $MergedModel `
    --outfile $GgufOutFile `
    --outtype f16

Write-Host ""
Write-Host "Conversion complete: $GgufOutFile"
Write-Host ""
Write-Host "Optional next step: quantize this file to reduce size (see deployment/README.md),"
Write-Host "or point deployment/Modelfile directly at this fp16 file and run:"
Write-Host "  ollama create truck-dispatch-ai -f deployment\Modelfile"
