@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "REPORT_DIR=reports"
set "REPORT_FILE=%REPORT_DIR%\ibkr_check.txt"

echo TradingBot IBKR paper read-only check
echo No orders will be submitted, modified, or cancelled.
echo.

if not exist "data" mkdir "data"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

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

echo Installing or verifying IBKR dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto dependency_error

"%PYTHON%" -m pip install -e ".[ibkr]"
if errorlevel 1 goto dependency_error

echo.
echo Connecting to local TWS paper API in read-only mode...
(
    "%PYTHON%" -m trading_bot.app ibkr-check --host 127.0.0.1 --port 7497 --client-id 15 --contracts "config\ibkr_swedish_contracts.json" --database "data\tradingbot.sqlite3"
) > "%REPORT_FILE%" 2>&1

echo.
echo IBKR CHECK COMPLETE
echo Report saved to: %REPORT_FILE%
echo.
pause
exit /b 0

:dependency_error
echo.
echo ERROR: Could not install or verify IBKR dependencies.
echo Check Python and internet access, then try again.
echo.
pause
exit /b 1
