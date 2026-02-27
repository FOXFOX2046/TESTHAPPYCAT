@echo off
REM Setup virtual environment for FoxGI WebApp
cd /d "%~dp0.."
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo success
