@echo off
rem Start the TrialMatch browser app (double-click). Run setup_app.bat once first.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo The app is not set up yet. Double-click setup_app.bat first.
  pause
  exit /b 1
)
echo Starting TrialMatch Screener... the browser opens automatically. Close this window to stop the app.
.venv\Scripts\python.exe -m app --open-browser %*
pause
