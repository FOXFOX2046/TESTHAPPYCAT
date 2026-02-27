@echo off
REM Build foxgi_web_app.zip excluding .venv, __pycache__, .git
cd /d "%~dp0.."
powershell -Command "Compress-Archive -Path app.py, requirements.txt, README.md, core, data, scripts, tests -DestinationPath foxgi_web_app.zip -Force"
echo Done.
