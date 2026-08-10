@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo Could not create .venv.
        echo Install Python from python.org and run setup_env.bat again.
        pause
        exit /b 1
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
if errorlevel 1 (
    echo.
    echo Could not upgrade pip in .venv.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements-portable.txt
if errorlevel 1 (
    echo.
    echo Could not install all required packages into .venv.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import h5py; print('Verified h5py', h5py.__version__, 'in', h5py.__file__)"
if errorlevel 1 (
    echo.
    echo h5py could not be imported from this .venv.
    echo If this project was copied from another PC, delete the copied .venv folder
    echo and run setup_env.bat again. Virtual environments cannot be copied between PCs.
    pause
    exit /b 1
)

echo.
echo Setup complete. Run run_gui.bat to start the GUI.
pause
