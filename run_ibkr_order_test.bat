@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "REPORT_DIR=reports"
set "REPORT_FILE=%REPORT_DIR%\ibkr_order_test.txt"

echo TradingBot IBKR PAPER order round-trip test
echo This can submit exactly one PAPER BUY and one PAPER SELL for ERIC-B.ST.
echo It is blocked by account, endpoint, contract, quantity, and explicit test-mode guards.
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
echo Running guarded PAPER order round-trip test...
(
    "%PYTHON%" -m trading_bot.app ibkr-order-test --enable-paper-order-test --host 127.0.0.1 --port 7497 --client-id 15 --symbol ERIC-B.ST --quantity 1 --timeout 60 --contracts "config\ibkr_swedish_contracts.json" --database "data\tradingbot.sqlite3"
) > "%REPORT_FILE%" 2>&1

if errorlevel 1 (
    echo.
    echo IBKR PAPER ORDER TEST FAILED OR WAS SAFELY BLOCKED
    echo Report saved to: %REPORT_FILE%
    echo Open the report for the exact guard failure or broker status.
    echo.
    pause
    exit /b 1
)

echo.
echo IBKR PAPER ORDER TEST COMPLETE
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
