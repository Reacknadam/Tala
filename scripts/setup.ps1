# TALA one-time setup (Windows / PowerShell)
# Creates a venv, installs backend + frontend deps, fetches model weights.
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "==> Python virtualenv" -ForegroundColor Cyan
python -m venv "$ROOT\backend\.venv"
& "$ROOT\backend\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$ROOT\backend\.venv\Scripts\python.exe" -m pip install -r "$ROOT\backend\requirements.txt"

Write-Host "==> Frontend dependencies" -ForegroundColor Cyan
Push-Location "$ROOT\frontend"
npm install
Pop-Location

Write-Host "==> Model weights" -ForegroundColor Cyan
& "$ROOT\backend\.venv\Scripts\python.exe" "$ROOT\scripts\download_models.py"

Write-Host "`nDone. Run scripts\dev.ps1 to start backend + frontend." -ForegroundColor Green
