@echo off
REM Run FoxGI Streamlit app
REM Change PORT if 8501 is in use: set PORT=8502
set PORT=8501
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
streamlit run app.py --server.port %PORT%
pause
