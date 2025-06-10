@echo off
REM start.bat — Ingest new docs and launch the FastAPI server

REM 1) (Optional) load environment variables from .env
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
  if not "%%A"=="#*" set "%%A=%%B"
)

REM 2) Ingest
echo ⏳ Ingesting documents...
python -m src.ingest

REM 3) Launch API
echo 🚀 Starting uvicorn on 0.0.0.0:8000
python -m uvicorn src.chatbot:app --host 0.0.0.0 --port 8000 --reload

pause
