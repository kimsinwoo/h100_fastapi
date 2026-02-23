@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -q -r requirements.txt
echo Starting Z-Image AI server on http://0.0.0.0:7000
uvicorn app.main:app --host 0.0.0.0 --port 7000
