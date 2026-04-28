@echo off
title Nedbank VolumeAI - Transaction Forecasting
echo.
echo  ============================================
echo   Nedbank VolumeAI - Transaction Forecasting
echo   Developed By Mashaba Hlulani Charles
echo  ============================================
echo.

:: Change to the directory where this script lives
cd /d "%~dp0"

:: Try to find Python (py launcher first, then python, then python3)
set PYTHON_CMD=
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=py
    goto FOUND_PYTHON
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python
    goto FOUND_PYTHON
)
where python3 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python3
    goto FOUND_PYTHON
)

echo [ERROR] Python is not installed or not in PATH.
echo Please install Python from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:FOUND_PYTHON
echo [INFO] Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [Setup] Creating virtual environment (first-time setup)...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo Falling back to running without venv...
        goto RUN_DIRECT
    )
    echo [Setup] Virtual environment created.
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo [Setup] Checking dependencies...
pip install --quiet -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some packages may have failed. Trying individual installs...
    pip install --quiet flask pandas numpy scikit-learn joblib pyarrow
)

echo.
echo [INFO] Starting Nedbank VolumeAI server...
echo [INFO] Open your browser to: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server.
echo.
python app.py
goto END

:RUN_DIRECT
echo.
echo [INFO] Starting without virtual environment...
echo [INFO] Open your browser to: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server.
echo.
%PYTHON_CMD% app.py

:END
echo.
echo Server stopped.
pause
