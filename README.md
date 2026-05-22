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

Large/binary packages such as `PyQt5`, `numpy`, `h5py`, `matplotlib`, and `scipy`
are installed by `pip` rather than copied into `vendor/`.

## First-Time Setup On Another Windows PC

1. Install Python 3.12 or another recent Python 3 version.
2. Copy this whole portable folder to the PC.
3. Double-click `setup_env.bat`.
4. Wait for the package installation to finish.
5. Double-click `run_gui.bat`.

If `setup_env.bat` cannot find `py`, install Python from python.org and make sure
the Python Launcher option is enabled.

## Opening Data

1. Start the GUI with `run_gui.bat`.
2. Set `Results root` to the ARTIQ results folder, for example:

   ```text
   Z:\artiqResults\lab1_bob
   ```

3. Click `Refresh` or wait automatic refresh.
4. Select a result file on the left.

If the selected file contains ndscan data, the center panel shows the ndscan plot
automatically. If it is not an ndscan file, the GUI falls back to the built-in
matplotlib plot area.

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
`nightly_monitor_GUI.py`, because the embedded ndscan plotting mode needs
`ndscan_gui_bridge/` and `vendor/`.

The receiver does not need to activate the original `artiq-oitg` environment.
