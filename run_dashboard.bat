@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo TradingBot local dashboard
echo This is research and paper-simulation only. It cannot trade real money.
echo.

if not exist "data" mkdir "data"
if not exist "reports" mkdir "reports"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    echo Creating local Python environment in .venv...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create a Python virtual environment.
        echo Install Python 3.11 or newer, then double-click this file again.
        echo.
        pause
        exit /b 1
    )
    set "PYTHON=.venv\Scripts\python.exe"
)

echo Installing or verifying dashboard dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto dependency_error

"%PYTHON%" -m pip install -e .
if errorlevel 1 goto dependency_error

echo.
echo Starting TradingBot dashboard at http://localhost:8000 ...
start "" "http://localhost:8000"
"%PYTHON%" -m trading_bot.app dashboard --host localhost --port 8000 --database "data\tradingbot.sqlite3" --current-cache-dir "data\current"
if errorlevel 1 goto dashboard_error

pause
exit /b 0

:dependency_error
echo.
echo ERROR: Could not install or verify required dependencies.
echo Check your Python installation, then try again.
echo.
pause
exit /b 1

:dashboard_error
echo.
echo ERROR: Dashboard failed to start.
echo Check that port 8000 is not already in use, then try again.
echo.
pause
exit /b 1
