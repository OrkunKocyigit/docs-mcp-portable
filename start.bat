@echo off
rem ============================================================================
rem  docs-portable launcher
rem  First run: unpacks a private Python 3.12 into .\runtime\python, then hands over
rem  to bootstrap.py, which installs dependencies, Node.js, docs-mcp-server,
rem  Chromium and the BGE-M3 model - all inside .\runtime.
rem  Every run: starts the embedding server, then docs-mcp-server.
rem  Usage: start.bat [run | setup | embedding-server | cli ...]
rem  Uses only tools built into Windows 10+ (curl.exe, tar.exe).
rem ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
title docs-portable
cd /d "%~dp0"

set "ROOT=%CD%"
set "RT=%ROOT%\runtime"
set "PY_DIR=%RT%\python"
set "PY=%PY_DIR%\python.exe"
set "PY_VERSION=3.12.10"

set "DOCS_PORTABLE_HOME=%ROOT%"
set "PYTHONHOME="
set "PYTHONPATH="
set "VIRTUAL_ENV="

rem ---- 1. Private Python 3.12 (official embeddable build) --------------------
if exist "%PY%" goto have_python
set "PY_ARCH=amd64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "PY_ARCH=arm64"
set "PY_ZIP=%RT%\python-embed.zip"
echo [setup] Downloading Python %PY_VERSION% (%PY_ARCH%)...
if not exist "%PY_DIR%" mkdir "%PY_DIR%"
curl.exe -L --fail --silent --show-error -o "%PY_ZIP%" "https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-embed-%PY_ARCH%.zip"
if errorlevel 1 goto fail_python
tar.exe -xf "%PY_ZIP%" -C "%PY_DIR%"
if errorlevel 1 goto fail_python
del "%PY_ZIP%" >nul 2>&1
if not exist "%PY%" goto fail_python
:have_python

rem ---- 2. Everything else is done by Python ----------------------------------
"%PY%" -X utf8 "%ROOT%\bootstrap.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fail_run
exit /b 0

:fail_python
echo [setup] Could not download or unpack Python. Check your internet connection and try again.
rmdir /s /q "%PY_DIR%" >nul 2>&1
goto fail_pause
:fail_run
echo.
echo [run] docs-portable exited with code %RC%.
:fail_pause
pause
exit /b 1
