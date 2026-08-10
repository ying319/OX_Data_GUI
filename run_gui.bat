@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run setup_env.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import h5py" >nul 2>&1
if errorlevel 1 (
    echo h5py is not available in this project's .venv.
    echo If this folder came from another PC, delete .venv and run setup_env.bat again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" OX_Data_GUI.py

if errorlevel 1 pause
