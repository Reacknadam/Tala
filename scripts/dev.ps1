# TALA dev launcher (Windows / PowerShell): backend on :8000, frontend on :5173
$ROOT = Split-Path -Parent $PSScriptRoot
$py = "$ROOT\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }  # fall back to global interpreter

Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" -WorkingDirectory "$ROOT\backend"
Write-Host "Backend  -> http://127.0.0.1:8000 (docs: /docs)" -ForegroundColor Cyan
Write-Host "Frontend -> http://localhost:5173" -ForegroundColor Cyan
Set-Location "$ROOT\frontend"
npm run dev
