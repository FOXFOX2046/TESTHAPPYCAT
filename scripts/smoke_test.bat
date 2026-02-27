@echo off
REM Run pytest smoke tests
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m pytest tests -q
pause
