@echo off
rem Abre a demo: 20x20 e 54x54 lado a lado (dois cliques neste arquivo)
cd /d "%~dp0"
set "PY=python"
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" v8\demo.py %*
if errorlevel 1 pause
