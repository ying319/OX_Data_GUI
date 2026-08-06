# Data GUI

This folder is a portable copy of the ARTIQ data GUI.

## What Is Included

- `OX_Data_GUI.py`: the main GUI.
- `ndscan_gui_bridge/`: glue code that embeds ndscan plots in the GUI.
- `vendor/`: local copies of the pure-Python ndscan-side packages:
  - `ndscan`
  - `pyqtgraph`
  - `qasync`
  - `sipyco`
  - `oitg`
- `requirements-portable.txt`: packages that should be installed into the local Python environment.
- `setup_env.bat`: creates a `.venv` and installs requirements.
- `run_gui.bat`: starts the GUI.
- `setup_env.sh`: creates a `.venv` and installs requirements on Linux.
- `run_gui.sh`: starts the GUI on Linux.
- `setup_env.command`: creates a `.venv` and installs requirements on macOS.
- `run_gui.command`: starts the GUI on macOS.

The environment setup installs these packages from `requirements-portable.txt`:

- `h5py`
- `matplotlib`
- `numpy`
- `PyQt5`
- `scipy`
- `lmfit`
- `allantools`
- `colorama`
- `statsmodels`

These packages are installed by `pip` rather than copied into `vendor/`.

## First-Time Setup On Another Windows PC

1. Install Python 3.12 or another recent Python 3 version from python.org.
2. Get this project folder onto the PC:
   - If using GitHub, click `Code` > `Download ZIP`, then unzip it.
   - Or use Git:

     ```powershell
     git clone https://github.com/ying319/portable-OX-Data-gui.git
     ```

   - Or copy the whole folder from another computer.
3. Open the project folder.
4. Double-click `setup_env.bat`.
5. Wait for the package installation to finish. This creates a local `.venv`
   folder and installs the required Python packages.
6. Double-click `run_gui.bat` to start the GUI.

After the first setup, you normally only need to double-click `run_gui.bat`.

If `setup_env.bat` cannot find `py`, install Python from python.org and make sure
the Python Launcher option is enabled.

## Running From PowerShell

If you prefer using PowerShell instead of double-clicking the batch files, open
PowerShell in this folder and run:

```powershell
.\setup_env.bat
.\run_gui.bat
```

## First-Time Setup On Linux

1. Install Python 3 and the Qt system libraries. On Ubuntu/Debian, run:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip libxcb-cursor0
   ```

   Other Linux distributions may use different package names, but you need
   Python 3, `venv`, `pip`, and the Qt/XCB runtime libraries used by PyQt5.

2. Get this project folder onto the computer:
   - If using GitHub, click `Code` > `Download ZIP`, then unzip it.
   - Or use Git:

     ```bash
     git clone https://github.com/ying319/portable-OX-Data-gui.git
     cd portable-OX-Data-gui
     ```

   - Or copy the whole folder from another computer.

3. Open a terminal in the project folder.
4. Make the helper scripts executable:

   ```bash
   chmod +x setup_env.sh run_gui.sh
   ```

5. Create a local virtual environment and install the Python packages:

   ```bash
   ./setup_env.sh
   ```

6. Start the GUI:

   ```bash
   ./run_gui.sh
   ```

After the first setup, you normally only need to run `./run_gui.sh` from the
project folder.

If the GUI fails to start with a Qt platform plugin error, install your
distribution's PyQt5/Qt XCB support packages. On Ubuntu/Debian, `libxcb-cursor0`
is the most common missing package.

## First-Time Setup On macOS

1. Install Python 3 from python.org or with Homebrew:

   ```bash
   brew install python
   ```

   If you do not use Homebrew, the installer from python.org is fine.

2. Get this project folder onto the Mac:
   - If using GitHub, click `Code` > `Download ZIP`, then unzip it.
   - Or use Git:

     ```bash
     git clone https://github.com/ying319/portable-OX-Data-gui.git
     cd portable-OX-Data-gui
     ```

   - Or copy the whole folder from another computer.

3. Open Terminal in the project folder and make the macOS helper files executable:

   ```bash
   chmod +x setup_env.command run_gui.command
   ```

4. Run the setup:
   - Double-click `setup_env.command`, or run:

     ```bash
     ./setup_env.command
     ```

5. Start the GUI:
   - Double-click `run_gui.command`, or run:

     ```bash
     ./run_gui.command
     ```

After the first setup, you normally only need to double-click `run_gui.command`.

If macOS says the file cannot be opened because it is from an unidentified
developer, right-click the `.command` file, choose `Open`, then confirm.

## Opening Data

1. Start the GUI with `run_gui.bat` on Windows, `./run_gui.sh` on Linux, or
   `run_gui.command` on macOS.
2. Set `Results root` to the ARTIQ results folder, for example:

   ```text
   Z:\artiqResults\lab1_bob
   ```

   On Linux or macOS this will usually be a mounted path, for example:

   ```text
   /mnt/artiqResults/lab1_bob
   /Volumes/artiqResults/lab1_bob
   ```

3. Click `Refresh` or wait automatic refresh.
4. Select a result file on the left.

If the selected file contains ndscan data, the center panel shows the ndscan plot
automatically. If it is not an ndscan file, the GUI falls back to the built-in
matplotlib plot area.

## Opening Logs

The `Logs` tab browses dated ARTIQ log files separately from HDF5 result files.
By default it opens:

```text
Z:\artiqResults\lab1_bob\log
```

Use `Browse` if the log folder is somewhere else. The loader scans the selected
folder for files with a `yyyy-mm-dd` date in the name and a log-style filename,
including names such as `2026-06-09.log`, `controller.2026-06-09.log`, and
`log.2026-06-09`.

- `Latest 50`: shows the 50 most recent dated log files across all dates.
- `Date`: when `Latest 50` is off, shows log files for the selected date.
- `Filter`: searches the selected log file's displayed entries.
- `1am-7am only`: shows entries timestamped from 01:00 up to before 07:00.

Multi-line log entries are grouped under the timestamped first line, and warning
or error lines are highlighted. Very large matching logs are previewed up to the
first 20,000 displayed lines so the GUI stays responsive.

## Plot Modes

- `Show ndscan Plot`: return to the embedded ndscan plot for the selected file.
- `Plot Selected`: plot the selected dataset(s) using the GUI's built-in matplotlib plotter.
- `Plot History`: plot scalar datasets across the currently visible result files.

## Filtering

- `1am-7am only`: shows result files whose recorded start time is from 01:00 up to
  before 07:00 to monitor the nightly automatic calibration.
- `Filter`: searches visible files by RID, class name, file name, and path.
- `Max`: limits how many recent files are loaded.

Refresh is cached, so repeated refreshes should be much faster when files have not
changed.

## Files To Share

Share the entire portable folder or the `portable_data_gui.zip` archive. Do not share only
`OX_Data_GUI.py`, because the embedded ndscan plotting mode needs
`ndscan_gui_bridge/` and `vendor/`.

The receiver does not need to activate the original `artiq-oitg` environment.
