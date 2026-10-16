@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.11 or 3.12 from python.org
  echo Make sure "Add Python to PATH" is selected.
  pause
  exit /b 1
)
py -3.11 -m venv .venv
if errorlevel 1 py -3.12 -m venv .venv
if errorlevel 1 (
  echo Could not create virtual environment. Install Python 3.11 or 3.12.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check the error above and your internet connection.
  pause
  exit /b 1
)
echo.
echo Installation complete. Run run_windows.bat
pause
