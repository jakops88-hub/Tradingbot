@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo TradingBot OpenAI configuration
echo This stores OPENAI_API_KEY for future double-click AI scans.
echo The key is not written to TradingBot reports.
echo.

for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$s=Read-Host 'Enter OPENAI_API_KEY' -AsSecureString; $b=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }"`) do set "OPENAI_API_KEY_VALUE=%%A"

if "%OPENAI_API_KEY_VALUE%"=="" (
    echo.
    echo ERROR: No API key was entered.
    echo.
    pause
    exit /b 1
)

set /p OPENAI_MODEL_VALUE=OpenAI model [press Enter for gpt-5.6-terra]: 
if "%OPENAI_MODEL_VALUE%"=="" set "OPENAI_MODEL_VALUE=gpt-5.6-terra"

setx OPENAI_API_KEY "%OPENAI_API_KEY_VALUE%" >nul
if errorlevel 1 goto config_error

setx OPENAI_MODEL "%OPENAI_MODEL_VALUE%" >nul
if errorlevel 1 goto config_error

set "OPENAI_API_KEY_VALUE="

echo.
echo OPENAI CONFIGURATION COMPLETE
echo OPENAI_MODEL: %OPENAI_MODEL_VALUE%
echo Close and reopen any command windows before running AI scans.
echo.
pause
exit /b 0

:config_error
set "OPENAI_API_KEY_VALUE="
echo.
echo ERROR: Could not save OpenAI environment variables.
echo Try running this file again, or set OPENAI_API_KEY manually in Windows.
echo.
pause
exit /b 1
