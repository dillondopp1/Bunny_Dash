@echo off
REM Double-click this file to start the pole vault editor on Windows.
cd /d "%~dp0"

set PY=
py -3 -c "import sys" >nul 2>&1 && set PY=py -3
if "%PY%"=="" python -c "import sys" >nul 2>&1 && set PY=python
if "%PY%"=="" (
  echo Python is not installed.
  echo Get it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
  echo then double-click this file again.
  pause
  exit /b 1
)

%PY% -c "import mediapipe, cv2, scipy" >nul 2>&1
if errorlevel 1 (
  echo Setting up for the first time. This takes a few minutes and only happens once.
  %PY% -m pip install --quiet --upgrade pip
  %PY% -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Setup failed. The message above says why.
    pause
    exit /b 1
  )
  echo Setup done.
)

echo.
echo Where do you want to use the editor?
echo   1^) Just this computer
echo   2^) This computer and my phone on the same wifi
set choice=1
set /p choice="Choose 1 or 2 [1]: "
echo.
if "%choice%"=="2" (
  %PY% tools\vault_app.py --lan
) else (
  %PY% tools\vault_app.py
)
pause
