@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "SYMBOL=VOLV-B.ST"
set "START_DATE=2018-01-01"
set "END_DATE=2026-01-01"
set "RISK=MEDIUM"
set "PORTFOLIO_CURRENCY=SEK"
set "PERCENTAGE_FEE=0.001"
set "FIXED_FEE=0"
set "SLIPPAGE=0.001"
set "DATA_DIR=data"
set "REPORT_DIR=reports"
set "DATA_FILE=%DATA_DIR%\%SYMBOL%_daily.csv"
set "REPORT_FILE=%REPORT_DIR%\latest_research.txt"

echo TradingBot one-click research
echo This runs a historical simulation only. It cannot trade real money.
echo.

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
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
echo Downloading %SYMBOL% daily data from %START_DATE% to %END_DATE%...
"%PYTHON%" -m trading_bot.app download "%SYMBOL%" --start "%START_DATE%" --end "%END_DATE%" --output-dir "%DATA_DIR%"
if errorlevel 1 goto download_error

echo.
echo Running EMA 20/50 research...
(
    echo TradingBot one-click research report
    echo Symbol: %SYMBOL%
    echo Date range: %START_DATE% to %END_DATE%
    echo Starting capital: 1000 SEK
    echo Risk profile: %RISK%
    echo Percentage fee: 0.1%%
    echo Fixed fee: %FIXED_FEE% SEK
    echo Slippage: 0.1%%
    echo Source data: %DATA_FILE%
    echo.
    "%PYTHON%" -m trading_bot.app research "%DATA_FILE%" --symbol "%SYMBOL%" --risk "%RISK%" --portfolio-currency "%PORTFOLIO_CURRENCY%" --percentage-fee "%PERCENTAGE_FEE%" --fixed-fee "%FIXED_FEE%" --slippage "%SLIPPAGE%"
) > "%REPORT_FILE%" 2>&1
if errorlevel 1 goto research_error

echo.
echo RESEARCH COMPLETE
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

:download_error
echo.
echo ERROR: Historical data download failed.
echo Possible causes: no internet connection, Yahoo Finance unavailable, or invalid ticker.
echo.
pause
exit /b 1

:research_error
echo.
echo ERROR: Research failed.
echo Details were saved to: %REPORT_FILE%
echo Open that file to read the error message.
echo.
pause
exit /b 1
