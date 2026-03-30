@echo off
echo Starting FrameSense Backend on 0.0.0.0:8000...
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
