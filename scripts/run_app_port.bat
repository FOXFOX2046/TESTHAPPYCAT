@echo off
REM Run FoxGI on alternate port (use when 8501 is busy)
REM Usage: run_app_port.bat [PORT]  e.g. run_app_port.bat 8502
set PORT=8502
if not "%~1"=="" set PORT=%~1
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
echo Starting on port %PORT%
streamlit run app.py --server.port %PORT%
pause
