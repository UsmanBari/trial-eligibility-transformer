@echo off
rem One-time setup of the browser app on Windows (double-click). Needs Python 3.10-3.12 and internet.
cd /d "%~dp0"
set PY=
for %%v in (3.12 3.11 3.10) do (
  if not defined PY ( py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v )
)
if not defined PY (
  echo Python 3.10, 3.11 or 3.12 was not found. Install Python 3.12 from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)
echo Using %PY%
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
.venv\Scripts\python.exe -m pip install -e . || goto :fail
.venv\Scripts\python.exe scripts\prepare_data.py
echo.
echo Setup finished. Double-click start_app.bat to run the app.
pause
exit /b 0
:fail
echo.
echo Setup failed. Scroll up for the first red error.
pause
exit /b 1
