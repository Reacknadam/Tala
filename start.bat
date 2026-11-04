@echo off
rem TALA launcher - double-click this file.
rem Starts the backend (:8000) and frontend (:5173) in separate windows.
setlocal
set "ROOT=%~dp0"

rem Prefer the backend venv if setup.ps1 created one, else global python.
set "PY=%ROOT%backend\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting TALA backend on http://127.0.0.1:8000 ...
start "TALA backend" cmd /k "cd /d "%ROOT%backend" && "%PY%" -m uvicorn app.main:app --reload --port 8000"

echo Starting TALA frontend on http://localhost:5173 ...
start "TALA frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

rem Give Vite a moment, then open the app.
timeout /t 4 /nobreak >nul
start http://localhost:5173
