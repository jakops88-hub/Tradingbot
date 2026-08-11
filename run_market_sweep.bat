@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "REPORT_DIR=reports"
set "REPORT_FILE=%REPORT_DIR%\market_sweep.txt"

echo TradingBot one-click Swedish market sweep
echo This runs a historical simulation only. It cannot trade real money.
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

echo Installing or verifying required dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto dependency_error

"%PYTHON%" -m pip install -e ".[market-data]"
if errorlevel 1 goto dependency_error

echo.
echo Running locked EMA 20/50 Swedish market sweep...
(
    echo TradingBot one-click market sweep report
    echo Symbols: config\swedish_large_caps.txt
    echo Period: 2018-01-01 to 2026-01-01
    echo Starting capital per instrument: 1000 SEK
    echo Risk profile: MEDIUM
    echo Percentage fee: 0.1%%
    echo Slippage: 0.1%%
    echo Adjustment policy: adjusted
    echo.
    "%PYTHON%" -m trading_bot.app market-sweep --symbols "config\swedish_large_caps.txt" --output-dir "data"
) > "%REPORT_FILE%" 2>&1
if errorlevel 1 goto sweep_error

echo.
echo MARKET SWEEP COMPLETE
echo Report saved to: %REPORT_FILE%
echo.
pause
exit /b 0

:dependency_error
echo.
echo ERROR: Could not install or verify required dependencies.
echo Check your internet connection and Python installation, then try again.
echo.
pause
exit /b 1

:sweep_error
echo.
echo ERROR: Market sweep failed.
echo Details were saved to: %REPORT_FILE%
echo Open that file to read the error message.
echo.
pause
exit /b 1
