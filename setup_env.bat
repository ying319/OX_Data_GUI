@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
)

".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 (
    echo.
    echo Could not install pip into .venv.
    echo Reinstall Python from python.org and include pip, then run setup_env.bat again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-portable.txt

echo.
echo Setup complete. Run run_gui.bat to start the GUI.
pause
