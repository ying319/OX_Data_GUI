# -*- coding: utf-8 -*-
"""Qt GUI for browsing nightly/monitor ARTIQ result files.

This viewer is intentionally generic: nightly.py schedules many different
experiments, so the useful data may be simple monitor scalars, ndscan arrays,
or analysis/fitting channels.  The GUI reads numeric datasets from ARTIQ HDF5
files and provides single-file plots plus history plots across many RIDs.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QDate, QSignalBlocker, QTimer, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ndscan_gui_bridge import NdscanPlotError, create_ndscan_plot_widget


DEFAULT_RESULTS_DIR = "Z:\\artiqResults\\lab1_bob"
DEFAULT_LOG_DIR = "Z:\\artiqResults\\lab1_bob\\log"
NUMERIC_KINDS = set("biufc")
NIGHTLY_START_HOUR = 1
NIGHTLY_END_HOUR = 7
SUMMARY_READ_WORKERS = 8
SCAN_WORKERS = 1
LOG_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
LOG_TIMESTAMP_RE = re.compile(
    r"^\[?(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):\d{2}:\d{2}(?:[.,]\d+)?\]?"
)
LOG_LEVEL_RE = re.compile(r"\b(CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG|TRACE)\b", re.IGNORECASE)
LOG_PREVIEW_LINES = 20_000
LOG_FILE_LIST_LIMIT = 50


@dataclass(frozen=True)
class ResultFile:
    path: Path
    rid: int | None
    start_time: float | None
    run_time: float | None
    class_name: str
    file_name: str


@dataclass(frozen=True)
class FileEntry:
    path: Path
    mtime: float
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class DatasetInfo:
    root: str
    key: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def label(self) -> str:
        return f"{self.root}/{self.key}"


@dataclass(frozen=True)
class LogFile:
    path: Path
    date: str
    mtime: float
    size: int


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_expid(raw: Any) -> dict[str, Any]:
    raw = _decode(raw)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(
            text.replace("false", "False")
            .replace("true", "True")
            .replace("null", "None")
        )
    except Exception:
        return {}


def rid_from_path(path: Path) -> int | None:
    stem = path.stem
    prefix = stem.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def format_time(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return ""
    try:
        return datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return f"{seconds:.3f}"


def is_in_nightly_time_span(seconds: float | None) -> bool:
    if seconds is None or not np.isfinite(seconds):
        return False
    try:
        hour = datetime.fromtimestamp(seconds).hour
    except Exception:
        return False
    return NIGHTLY_START_HOUR <= hour < NIGHTLY_END_HOUR


def path_mtime_in_nightly_time_span(entry: FileEntry) -> bool:
    try:
        hour = datetime.fromtimestamp(entry.mtime).hour
    except Exception:
        return False
    return NIGHTLY_START_HOUR <= hour < NIGHTLY_END_HOUR


def summarise_value(value: Any, max_items: int = 8) -> str:
    value = _decode(value)
    arr = np.asarray(value)
    if arr.shape == ():
        return str(_decode(arr.item()))
    flat = arr.ravel()
    preview = ", ".join(str(_decode(v)) for v in flat[:max_items])
    if flat.size > max_items:
        preview += ", ..."
    return f"shape={arr.shape}, [{preview}]"


def is_numeric_dataset(dset: h5py.Dataset) -> bool:
    return getattr(dset.dtype, "kind", "") in NUMERIC_KINDS


def iter_numeric_datasets(
    h5: h5py.File, roots: tuple[str, ...] = ("datasets", "archive")
) -> list[DatasetInfo]:
    infos: list[DatasetInfo] = []
    for root in roots:
        if root not in h5:
            continue

        def visit(name: str, obj: h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset) and is_numeric_dataset(obj):
                infos.append(DatasetInfo(root, name, tuple(obj.shape), str(obj.dtype)))

        h5[root].visititems(visit)
    return sorted(infos, key=lambda d: d.label.lower())


def iter_archive_rows(h5: h5py.File) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if "archive" not in h5:
        return rows

    def visit(name: str, obj: h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        shape = "scalar" if obj.shape == () else "x".join(map(str, obj.shape))
        rows.append((name, f"{shape}, {obj.dtype}", summarise_value(obj[()])))

    h5["archive"].visititems(visit)
    return sorted(rows, key=lambda row: row[0].lower())


def read_dataset(path: Path, info: DatasetInfo) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[info.root][info.key][()])


def read_file_summary(path: Path) -> ResultFile:
    rid = rid_from_path(path)
    start_time = None
    run_time = None
    class_name = ""
    exp_file = ""
    try:
        with h5py.File(path, "r") as h5:
            if "start_time" in h5:
                start_time = float(_decode(h5["start_time"][()]))
            if "run_time" in h5:
                run_time = float(_decode(h5["run_time"][()]))
            expid = parse_expid(h5["expid"][()] if "expid" in h5 else "")
            class_name = str(expid.get("class_name", ""))
            exp_file = str(expid.get("file", ""))
    except Exception:
        pass
    return ResultFile(path, rid, start_time, run_time, class_name, exp_file)


def path_is_available(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def default_results_root() -> tuple[str, bool]:
    default_root = Path(DEFAULT_RESULTS_DIR)
    return str(default_root), not path_is_available(default_root)


class PlotCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 5))
        super().__init__(self.figure)
        self.reset_layout()

    @property
    def ax(self):
        if not self.figure.axes:
            return self.figure.add_subplot(111)
        return self.figure.axes[0]

    def clear(self) -> None:
        self.figure.clear()
        self.figure.add_subplot(111)
        self.reset_layout()

    def reset_layout(self) -> None:
        self.figure.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.9)


class NightlyMonitorGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Data GUI")
        self.resize(1300, 820)

        self.files: list[ResultFile] = []
        self.file_entries: list[FileEntry] = []
        self.file_entries_root: Path | None = None
        self.visible_entries: dict[Path, FileEntry] = {}
        self.summary_cache: dict[Path, tuple[int, int, ResultFile]] = {}
        self.summary_executor = ThreadPoolExecutor(max_workers=SUMMARY_READ_WORKERS)
        self.pending_summary_futures: dict[Future, tuple[int, FileEntry]] = {}
        self.summary_generation = 0
        self.scan_executor = ThreadPoolExecutor(max_workers=SCAN_WORKERS)
        self.pending_scan_future: Future | None = None
        self.scan_generation = 0
        self.loading_file = False
        self.current_infos: dict[str, DatasetInfo] = {}
        self.current_file: ResultFile | None = None
        self.ndscan_widget: QWidget | None = None
        self.ndscan_h5_file: h5py.File | None = None
        self.log_files: list[LogFile] = []

        initial_root, used_home_fallback = default_results_root()
        self.path_edit = QLineEdit(initial_root)
        self.path_edit.setMaximumWidth(520)
        self.browse_button = QPushButton("Browse")
        self.refresh_button = QPushButton("Refresh")
        self.path_status = QLabel()
        self.path_status.setMinimumWidth(180)
        self.path_status.setStyleSheet("color: #9a5a00;")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by class, file, dataset, RID...")
        self.max_files = QComboBox()
        self.max_files.addItems(["50", "100", "200", "300", "500", "1000", "2000", "5000", "All"])
        self.max_files.setCurrentText("100")
        self.max_label = QLabel("Max")
        self.max_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.max_label.setMaximumWidth(28)
        self.monitor_only = QCheckBox("1am-7am only")
        self.monitor_only.setToolTip(
            "Show result files started from 1am up to before 7am."
        )
        self.monitor_only.setChecked(False)

        self.file_list = QListWidget()
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.archive_table = QTableWidget(0, 3)
        self.archive_table.setHorizontalHeaderLabels(["Name", "Type", "Value"])
        self.archive_table.horizontalHeader().setStretchLastSection(True)
        self.archive_table.setAlternatingRowColors(True)
        self.x_combo = QComboBox()
        self.x_combo.addItem("Index / scalar history")
        self.ndscan_plot_button = QPushButton("Show ndscan Plot")
        self.ndscan_plot_button.setEnabled(False)
        self.plot_button = QPushButton("Plot Selected")
        self.history_button = QPushButton("Plot History")
        self.auto_fit_curves = QCheckBox("Auto overlay fit curves")
        self.auto_fit_curves.setChecked(True)

        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.fit_table = QTableWidget(0, 3)
        self.fit_table.setHorizontalHeaderLabels(["Source", "Name", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.raw_summary = QLabel("Select a dataset to inspect raw values.")
        self.raw_table = QTableWidget(0, 0)
        self.raw_table.setAlternatingRowColors(True)
        self.export_raw_button = QPushButton("Export CSV")

        self.log_path_edit = QLineEdit(DEFAULT_LOG_DIR)
        self.log_path_edit.setMaximumWidth(520)
        self.log_browse_button = QPushButton("Browse")
        self.log_refresh_button = QPushButton("Refresh")
        self.log_date_edit = QDateEdit()
        self.log_date_edit.setCalendarPopup(True)
        self.log_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.log_date_edit.setDate(QDate.currentDate())
        self.log_latest_only = QCheckBox("Latest 50")
        self.log_latest_only.setChecked(True)
        self.log_latest_only.setToolTip("Show the most recent 50 log files across all dates.")
        self.log_filter_edit = QLineEdit()
        self.log_filter_edit.setPlaceholderText("Filter log text...")
        self.log_time_filter = QCheckBox("1am-7am only")
        self.log_time_filter.setToolTip("Show log entries timestamped from 01:00 up to before 07:00.")
        self.log_status = QLabel()
        self.log_status.setMinimumWidth(220)
        self.log_status.setStyleSheet("color: #9a5a00;")
        self.log_file_list = QListWidget()
        self.log_text = QTreeWidget()
        self.log_text.setHeaderHidden(True)
        self.log_text.setUniformRowHeights(True)
        self.log_text.setRootIsDecorated(True)
        self.log_text.setAlternatingRowColors(True)
        log_font = QFont("Consolas")
        log_font.setStyleHint(QFont.Monospace)
        self.log_text.setFont(log_font)

        self.canvas = PlotCanvas()
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.canvas)
        self.summary_timer = QTimer(self)
        self.summary_timer.setInterval(100)
        self.summary_timer.timeout.connect(self.poll_summary_reads)
        self.scan_timer = QTimer(self)
        self.scan_timer.setInterval(100)
        self.scan_timer.timeout.connect(self.poll_file_scan)

        self._build_layout()
        self._connect()
        if used_home_fallback:
            self.show_path_status(
                "Default results path is unavailable; Browse will open from your home folder."
            )

    def _build_layout(self) -> None:
        top = QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(QLabel("Results root"))
        top.addWidget(self.path_edit)
        top.addWidget(self.browse_button)
        top.addWidget(self.refresh_button)
        top.addWidget(self.path_status)
        top.addWidget(self.max_label)
        top.addWidget(self.max_files)
        top.addWidget(self.monitor_only)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter"))
        filter_row.addWidget(self.filter_edit)

        files_box = QGroupBox("Result Files")
        files_layout = QVBoxLayout(files_box)
        files_layout.addLayout(filter_row)
        files_layout.addWidget(self.file_list)

        data_box = QGroupBox("Result Datasets (Plot)")
        data_layout = QVBoxLayout(data_box)
        data_layout.addWidget(self.dataset_list)
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X"))
        x_row.addWidget(self.x_combo, 1)
        data_layout.addLayout(x_row)
        button_row = QHBoxLayout()
        button_row.addWidget(self.ndscan_plot_button)
        button_row.addWidget(self.plot_button)
        button_row.addWidget(self.history_button)
        data_layout.addLayout(button_row)
        data_layout.addWidget(self.auto_fit_curves)

        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(files_box)
        left_split.addWidget(data_box)

        archive_box = QGroupBox("Archive Parameters")
        archive_layout = QVBoxLayout(archive_box)
        archive_layout.addWidget(self.archive_table)
        left_split.addWidget(archive_box)
        left_split.setSizes([330, 260, 220])

        details_split = QSplitter(Qt.Vertical)
        details_split.addWidget(self.meta_text)

        fit_box = QGroupBox("Fit-Related Results")
        fit_layout = QVBoxLayout(fit_box)
        fit_layout.addWidget(self.fit_table)
        details_split.addWidget(fit_box)

        raw_box = QGroupBox("Raw Data")
        raw_layout = QVBoxLayout(raw_box)
        raw_layout.addWidget(self.raw_summary)
        raw_layout.addWidget(self.raw_table)
        raw_layout.addWidget(self.export_raw_button)
        details_split.addWidget(raw_box)
        details_split.setSizes([190, 190, 320])

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left_split)
        main_split.addWidget(self.plot_stack)
        main_split.addWidget(details_split)
        main_split.setSizes([360, 650, 290])

        data_tab = QWidget()
        data_layout_root = QVBoxLayout(data_tab)
        data_layout_root.addLayout(top)
        data_layout_root.addWidget(main_split, 1)

        log_tab = self.build_log_tab()

        tabs = QTabWidget()
        tabs.addTab(data_tab, "Data")
        tabs.addTab(log_tab, "Logs")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(tabs)
        self.setCentralWidget(root)

    def build_log_tab(self) -> QWidget:
        top = QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(QLabel("Log root"))
        top.addWidget(self.log_path_edit)
        top.addWidget(self.log_browse_button)
        top.addWidget(self.log_refresh_button)
        top.addWidget(QLabel("Date"))
        top.addWidget(self.log_date_edit)
        top.addWidget(self.log_latest_only)
        top.addWidget(self.log_status)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter"))
        filter_row.addWidget(self.log_filter_edit)
        filter_row.addWidget(self.log_time_filter)

        list_box = QGroupBox("Log Files")
        list_layout = QVBoxLayout(list_box)
        list_layout.addWidget(self.log_file_list)

        content_box = QGroupBox("Log Content")
        content_layout = QVBoxLayout(content_box)
        content_layout.addLayout(filter_row)
        content_layout.addWidget(self.log_text)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(list_box)
        split.addWidget(content_box)
        split.setSizes([320, 900])

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        return tab

    def _connect(self) -> None:
        self.browse_button.clicked.connect(self.choose_root)
        self.refresh_button.clicked.connect(lambda: self.refresh_files(force_scan=True))
        self.filter_edit.textChanged.connect(lambda _text: self.populate_file_list())
        self.monitor_only.stateChanged.connect(lambda _state: self.refresh_files(force_scan=False))
        self.max_files.currentTextChanged.connect(lambda _text: self.refresh_files(force_scan=False))
        self.file_list.currentItemChanged.connect(self.file_selected)
        self.dataset_list.itemSelectionChanged.connect(self.dataset_selection_changed)
        self.dataset_list.itemDoubleClicked.connect(lambda _item: self.plot_selected())
        self.ndscan_plot_button.clicked.connect(self.show_current_ndscan_plot)
        self.plot_button.clicked.connect(self.plot_selected)
        self.history_button.clicked.connect(self.plot_history)
        self.export_raw_button.clicked.connect(self.export_raw_csv)
        self.log_browse_button.clicked.connect(self.choose_log_root)
        self.log_refresh_button.clicked.connect(self.refresh_logs)
        self.log_date_edit.dateChanged.connect(lambda _date: self.populate_log_file_list())
        self.log_latest_only.stateChanged.connect(lambda _state: self.populate_log_file_list())
        self.log_filter_edit.textChanged.connect(lambda _text: self.load_selected_log())
        self.log_time_filter.stateChanged.connect(lambda _state: self.load_selected_log())
        self.log_file_list.currentItemChanged.connect(lambda current, _previous: self.log_file_selected(current))
        QTimer.singleShot(0, self.refresh_logs)

    def choose_log_root(self) -> None:
        start_dir = Path(self.log_path_edit.text()).expanduser()
        if not path_is_available(start_dir):
            start_dir = Path.home()
            self.show_log_status("Current path is invalid; browse opened from your home folder.")
        directory = QFileDialog.getExistingDirectory(
            self, "Select ARTIQ log directory", str(start_dir)
        )
        if directory:
            self.log_path_edit.setText(directory)
            self.refresh_logs()

    def show_log_status(self, message: str) -> None:
        self.log_status.setText(message)
        self.log_status.setToolTip(message)

    def refresh_logs(self) -> None:
        root = Path(self.log_path_edit.text()).expanduser()
        if not path_is_available(root):
            self.log_files = []
            self.log_file_list.clear()
            self.log_text.clear()
            self.show_log_status("Log folder not found.")
            return

        self.log_files = self.collect_log_files(root)
        if not self.log_files:
            self.log_file_list.clear()
            self.log_text.clear()
            self.show_log_status("No dated .log files found.")
            return

        selected_date = self.log_date_edit.date().toString("yyyy-MM-dd")
        available_dates = sorted({log_file.date for log_file in self.log_files})
        if not self.log_latest_only.isChecked() and selected_date not in available_dates:
            blocker = QSignalBlocker(self.log_date_edit)
            self.log_date_edit.setDate(QDate.fromString(available_dates[-1], "yyyy-MM-dd"))
            del blocker
        self.populate_log_file_list()

    def collect_log_files(self, root: Path) -> list[LogFile]:
        log_files: list[LogFile] = []
        try:
            with os.scandir(root) as scan:
                for item in scan:
                    if not item.is_file() or item.name.startswith("."):
                        continue
                    match = LOG_DATE_RE.search(item.name)
                    if match is None:
                        continue
                    lower_name = item.name.lower()
                    if not (
                        lower_name.endswith(".log")
                        or lower_name == f"log.{match.group(1)}"
                        or lower_name.endswith(f".{match.group(1)}")
                    ):
                        continue
                    try:
                        stat = item.stat()
                    except OSError:
                        continue
                    log_files.append(
                        LogFile(Path(item.path), match.group(1), stat.st_mtime, stat.st_size)
                    )
        except OSError as exc:
            self.show_log_status(f"Could not scan log folder: {exc}")
            return []
        return sorted(log_files, key=lambda log_file: (log_file.date, log_file.path.name), reverse=True)

    def populate_log_file_list(self) -> None:
        selected_date = self.log_date_edit.date().toString("yyyy-MM-dd")
        if self.log_latest_only.isChecked():
            matching = self.log_files[:LOG_FILE_LIST_LIMIT]
        else:
            matching = [
                log_file for log_file in self.log_files if log_file.date == selected_date
            ][:LOG_FILE_LIST_LIMIT]
        blocker = QSignalBlocker(self.log_file_list)
        self.log_file_list.clear()
        for log_file in matching:
            size_text = self.format_size(log_file.size)
            item = QListWidgetItem(f"{log_file.date}  {log_file.path.name} ({size_text})")
            item.setData(Qt.UserRole, log_file)
            self.log_file_list.addItem(item)
        del blocker

        if matching:
            self.log_file_list.setCurrentRow(0)
            if self.log_latest_only.isChecked():
                self.show_log_status(f"Showing latest {len(matching)} log file(s).")
            else:
                self.show_log_status(f"{len(matching)} log file(s) for {selected_date}.")
        else:
            self.log_text.clear()
            if self.log_latest_only.isChecked():
                self.show_log_status("No log files found.")
            else:
                self.show_log_status(f"No log files for {selected_date}.")

    def log_file_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            self.log_text.clear()
            return
        self.load_selected_log()

    def load_selected_log(self) -> None:
        item = self.log_file_list.currentItem()
        if item is None:
            return
        log_file = item.data(Qt.UserRole)
        if not isinstance(log_file, LogFile):
            return

        filter_text = self.log_filter_edit.text().strip().lower()
        time_filter_enabled = self.log_time_filter.isChecked()
        self.log_text.clear()
        shown = 0
        total = 0
        matched = 0
        current_title = ""
        current_children: list[str] = []
        current_in_time_span = not time_filter_enabled

        def flush_current() -> None:
            nonlocal current_title, current_children, current_in_time_span, shown, matched
            if not current_title:
                return
            if time_filter_enabled and not current_in_time_span:
                current_title = ""
                current_children = []
                return
            if filter_text:
                title_matches = filter_text in current_title.lower()
                matching_children = [
                    child for child in current_children if filter_text in child.lower()
                ]
                if not title_matches and not matching_children:
                    current_title = ""
                    current_children = []
                    return
                display_children = current_children if title_matches else matching_children
            else:
                display_children = current_children

            matched += 1 + len(display_children)
            if shown < LOG_PREVIEW_LINES:
                remaining = LOG_PREVIEW_LINES - shown
                children_to_show = display_children[: max(0, remaining - 1)]
                self.add_log_tree_entry(current_title, children_to_show)
                shown += 1 + len(children_to_show)
            current_title = ""
            current_children = []

        try:
            with log_file.path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    total += 1
                    text = line.rstrip("\n\r")
                    timestamp_match = LOG_TIMESTAMP_RE.match(text)
                    if timestamp_match is not None:
                        flush_current()
                        hour = int(timestamp_match.group(2))
                        current_title = text
                        current_children = []
                        current_in_time_span = NIGHTLY_START_HOUR <= hour < NIGHTLY_END_HOUR
                    elif current_title:
                        current_children.append(text)
                    else:
                        current_title = text
                        current_children = []
                        current_in_time_span = not time_filter_enabled
                flush_current()
        except OSError as exc:
            self.show_log_message(f"Could not read log file: {exc}", "error")
            self.show_log_status("Could not read selected log file.")
            return

        if shown == 0:
            self.show_log_message("No log entries match the current filters.", "debug")
        elif shown < matched:
            self.show_log_message(
                f"Only the first {LOG_PREVIEW_LINES:,} matching lines are shown.",
                "debug",
            )
        if filter_text:
            scope = "1am-7am " if time_filter_enabled else ""
            self.show_log_status(
                f"Showing {shown:,} of {matched:,} {scope}matching line(s) in {log_file.path.name}."
            )
        elif time_filter_enabled:
            self.show_log_status(
                f"Showing {shown:,} 1am-7am line(s) from {log_file.path.name}."
            )
        else:
            self.show_log_status(f"Showing {shown:,} of {total:,} line(s) in {log_file.path.name}.")

    def add_log_tree_entry(self, title: str, children: list[str]) -> None:
        level = self.log_line_level(title)
        parent = QTreeWidgetItem([title or " "])
        self.apply_log_item_style(parent, level)
        for child in children:
            child_item = QTreeWidgetItem([child or " "])
            self.apply_log_item_style(child_item, level)
            parent.addChild(child_item)
        self.log_text.addTopLevelItem(parent)

    def show_log_message(self, message: str, level: str = "plain") -> None:
        item = QTreeWidgetItem([message])
        self.apply_log_item_style(item, level)
        self.log_text.addTopLevelItem(item)

    def log_line_level(self, line: str) -> str:
        match = LOG_LEVEL_RE.search(line)
        if match is None:
            return "plain"
        level = match.group(1).upper()
        if level in {"CRITICAL", "ERROR"}:
            return "error"
        if level in {"WARNING", "WARN"}:
            return "warning"
        if level == "INFO":
            return "info"
        return "debug"

    def apply_log_item_style(self, item: QTreeWidgetItem, level: str) -> None:
        colors = {
            "plain": (QColor("#242424"), QColor("#fbfbf8")),
            "info": (QColor("#1f6f8b"), QColor("#eef8fc")),
            "warning": (QColor("#8a5a00"), QColor("#fff6d8")),
            "error": (QColor("#a52626"), QColor("#ffe8e8")),
            "debug": (QColor("#6b7280"), QColor("#f1f3f5")),
        }
        foreground, background = colors.get(level, colors["plain"])
        item.setForeground(0, foreground)
        item.setBackground(0, background)
        if level == "error":
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)

    def format_size(self, size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024 or unit == "GB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
            value /= 1024
        return f"{size} B"

    def choose_root(self) -> None:
        start_dir = Path(self.path_edit.text()).expanduser()
        if not path_is_available(start_dir):
            start_dir = Path.home()
            self.show_path_status("Current path is invalid; browse opened from your home folder.")
        directory = QFileDialog.getExistingDirectory(
            self, "Select ARTIQ results directory", str(start_dir)
        )
        if directory:
            self.path_edit.setText(directory)
            self.update_path_status(Path(directory))
            self.refresh_files(force_scan=True)

    def refresh_files(self, force_scan: bool = False) -> None:
        root = Path(self.path_edit.text()).expanduser()
        if not path_is_available(root):
            self.show_path_status("Current path is invalid. Choose an existing results folder.")
            QMessageBox.warning(self, "Missing directory", f"Directory not found:\n{root}")
            return
        self.update_path_status(root)
        max_text = self.max_files.currentText()
        max_count = None if max_text == "All" else int(max_text)
        root_changed = root != self.file_entries_root
        if root_changed:
            self.current_file = None
        if force_scan or root_changed or not self.file_entries:
            self.start_file_scan(root, max_count)
            return
        self.apply_file_entries(self.file_entries, max_count)

    def start_file_scan(self, root: Path, max_count: int | None) -> None:
        self.cancel_pending_summary_reads()
        self.scan_generation += 1
        if self.pending_scan_future is not None:
            self.pending_scan_future.cancel()
        self.file_entries_root = root
        self.file_entries = []
        self.visible_entries = {}
        self.files = []
        self.populate_file_list(auto_load_selected=False)
        self.show_path_status("Scanning result files...")
        generation = self.scan_generation
        scan_max_count = None if self.monitor_only.isChecked() else max_count
        self.pending_scan_future = self.scan_executor.submit(
            self.collect_file_entries, root, scan_max_count
        )
        self.pending_scan_future.generation = generation
        self.pending_scan_future.max_count = max_count
        self.scan_timer.start()

    def poll_file_scan(self) -> None:
        future = self.pending_scan_future
        if future is None or not future.done():
            return
        self.pending_scan_future = None
        self.scan_timer.stop()
        if future.cancelled() or getattr(future, "generation", None) != self.scan_generation:
            return
        try:
            entries = future.result()
        except Exception as exc:
            self.show_path_status(f"Could not scan result files: {exc}")
            return
        self.file_entries = entries
        self.update_path_status(self.file_entries_root or Path(self.path_edit.text()).expanduser())
        self.apply_file_entries(entries, getattr(future, "max_count", None))

    def apply_file_entries(self, entries: list[FileEntry], max_count: int | None) -> None:
        if self.monitor_only.isChecked():
            entries = [entry for entry in entries if path_mtime_in_nightly_time_span(entry)]
        entries.sort(key=lambda entry: entry.mtime, reverse=True)
        if max_count is not None:
            entries = entries[:max_count]
        self.summary_generation += 1
        self.cancel_pending_summary_reads()
        self.visible_entries = {entry.path: entry for entry in entries}
        self.files = [self.cached_or_placeholder_summary(entry) for entry in entries]
        self.populate_file_list(auto_load_selected=False)
        self.schedule_summary_reads(entries)

    def cancel_pending_summary_reads(self) -> None:
        for future in self.pending_summary_futures:
            future.cancel()
        self.pending_summary_futures.clear()
        self.summary_timer.stop()

    def show_path_status(self, message: str) -> None:
        self.path_status.setText(message)
        self.path_status.setToolTip(message)

    def update_path_status(self, root: Path) -> None:
        if path_is_available(root):
            self.show_path_status("")
        else:
            self.show_path_status("Current path is invalid. Choose an existing results folder.")

    def collect_file_entries(self, root: Path, max_count: int | None = None) -> list[FileEntry]:
        entries = self.collect_direct_file_entries(root, max_count)
        if entries:
            return entries

        entries = []
        for path in root.rglob("*.h5"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(FileEntry(path, stat.st_mtime, stat.st_mtime_ns, stat.st_size))
        return entries

    def collect_direct_file_entries(self, root: Path, max_count: int | None = None) -> list[FileEntry]:
        entries = []
        candidates: list[tuple[int, str, str]] = []
        try:
            with os.scandir(root) as scan:
                for item in scan:
                    if not item.is_file() or not item.name.lower().endswith(".h5"):
                        continue
                    candidates.append((rid_from_path(Path(item.name)) or -1, item.name, item.path))
        except OSError:
            return []

        if max_count is not None and len(candidates) > max_count * 3:
            candidates.sort(reverse=True)
            candidates = candidates[: max_count * 3]

        for _rid, _name, path_text in candidates:
            path = Path(path_text)
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(FileEntry(path, stat.st_mtime, stat.st_mtime_ns, stat.st_size))
        return entries

    def cached_or_placeholder_summary(self, entry: FileEntry) -> ResultFile:
        cached = self.summary_cache.get(entry.path)
        if cached is not None and cached[0] == entry.mtime_ns and cached[1] == entry.size:
            return cached[2]
        return ResultFile(entry.path, rid_from_path(entry.path), entry.mtime, None, "", "")

    def schedule_summary_reads(self, entries: list[FileEntry]) -> None:
        generation = self.summary_generation
        for entry in entries:
            cached = self.summary_cache.get(entry.path)
            if cached is not None and cached[0] == entry.mtime_ns and cached[1] == entry.size:
                continue
            future = self.summary_executor.submit(read_file_summary, entry.path)
            self.pending_summary_futures[future] = (generation, entry)
        if self.pending_summary_futures:
            self.summary_timer.start()

    def poll_summary_reads(self) -> None:
        done = [future for future in self.pending_summary_futures if future.done()]
        if not done:
            return
        changed = False
        for future in done:
            generation, entry = self.pending_summary_futures.pop(future)
            if generation != self.summary_generation or future.cancelled():
                continue
            try:
                summary = future.result()
            except Exception:
                summary = ResultFile(entry.path, rid_from_path(entry.path), entry.mtime, None, "", "")
            self.summary_cache[entry.path] = (entry.mtime_ns, entry.size, summary)
            changed = self.replace_file_summary(summary) or changed
        if changed:
            self.populate_file_list(auto_load_selected=False)
        if not self.pending_summary_futures:
            self.summary_timer.stop()

    def replace_file_summary(self, summary: ResultFile) -> bool:
        for idx, result in enumerate(self.files):
            if result.path == summary.path:
                self.files[idx] = summary
                if self.current_file is not None and self.current_file.path == summary.path:
                    self.current_file = summary
                return True
        return False

    def ensure_file_summary(self, result: ResultFile) -> ResultFile:
        entry = self.visible_entries.get(result.path)
        if entry is None:
            return result
        cached = self.summary_cache.get(result.path)
        if cached is not None and cached[0] == entry.mtime_ns and cached[1] == entry.size:
            summary = cached[2]
        else:
            summary = read_file_summary(result.path)
            self.summary_cache[result.path] = (entry.mtime_ns, entry.size, summary)
        self.replace_file_summary(summary)
        return summary

    def read_file_summaries(self, entries: list[FileEntry]) -> list[ResultFile]:
        summaries: list[ResultFile | None] = [None] * len(entries)
        missing: list[tuple[int, FileEntry]] = []

        for idx, entry in enumerate(entries):
            cached = self.summary_cache.get(entry.path)
            if cached is not None and cached[0] == entry.mtime_ns and cached[1] == entry.size:
                summaries[idx] = cached[2]
            else:
                missing.append((idx, entry))

        if missing:
            workers = min(SUMMARY_READ_WORKERS, len(missing), (os.cpu_count() or 1) + 4)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(read_file_summary, entry.path): (idx, entry)
                    for idx, entry in missing
                }
                for future in as_completed(futures):
                    idx, entry = futures[future]
                    try:
                        summary = future.result()
                    except Exception:
                        summary = ResultFile(entry.path, rid_from_path(entry.path), None, None, "", "")
                    self.summary_cache[entry.path] = (entry.mtime_ns, entry.size, summary)
                    summaries[idx] = summary

        return [summary for summary in summaries if summary is not None]

    def _file_matches_filter(self, result: ResultFile, query: str) -> bool:
        if self.monitor_only.isChecked() and not is_in_nightly_time_span(result.start_time):
            return False
        if not query:
            return True
        hay = (
            f"{result.rid} {result.class_name} {result.file_name} "
            f"{result.path.name} {result.path.parent}"
        ).lower()
        return all(part in hay for part in query.split())

    def populate_file_list(self, auto_load_selected: bool = True) -> None:
        query = self.filter_edit.text().strip().lower()
        selected_path = self.current_file.path if self.current_file is not None else None
        selected_row = -1
        blocker = QSignalBlocker(self.file_list)
        self.file_list.clear()
        for result in self.files:
            if not self._file_matches_filter(result, query):
                continue
            item = QListWidgetItem(self.format_file_label(result))
            item.setData(Qt.UserRole, result)
            item.setToolTip(str(result.path))
            self.file_list.addItem(item)
            if selected_path is not None and result.path == selected_path:
                selected_row = self.file_list.count() - 1
        if self.file_list.count() and (auto_load_selected or selected_row >= 0):
            self.file_list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        del blocker
        current = self.file_list.currentItem()
        if current is None:
            self.current_file = None
            return
        result = current.data(Qt.UserRole)
        if auto_load_selected and isinstance(result, ResultFile) and (
            self.current_file is None or result.path != self.current_file.path
        ):
            self.file_selected(current)

    def file_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        result = current.data(Qt.UserRole)
        if not isinstance(result, ResultFile):
            return
        self.show_path_status(f"Loading {result.path.name}...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = self.ensure_file_summary(result)
            current.setData(Qt.UserRole, result)
            current.setText(self.format_file_label(result))
            self.current_file = result
            self.ndscan_plot_button.setEnabled(True)
            self.load_file(result)
        finally:
            QApplication.restoreOverrideCursor()
            self.update_path_status(Path(self.path_edit.text()).expanduser())

    def format_file_label(self, result: ResultFile) -> str:
        rid = "" if result.rid is None else str(result.rid)
        when = format_time(result.start_time)
        return f"{rid:>9}  {when}  {result.class_name or result.path.name}"

    def load_file(self, result: ResultFile) -> None:
        self.close_ndscan_plot()
        self.show_matplotlib_plot()
        self.loading_file = True
        self.dataset_list.clear()
        self.archive_table.setRowCount(0)
        self.clear_raw_table()
        self.x_combo.clear()
        self.x_combo.addItem("Index / scalar history")
        self.current_infos = {}
        try:
            with h5py.File(result.path, "r") as h5:
                infos = iter_numeric_datasets(h5, roots=("datasets",))
                archive_rows = iter_archive_rows(h5)
                meta = self.describe_file(h5, result)
                fit_rows = self.collect_fit_rows(h5)
        except Exception as exc:
            self.loading_file = False
            QMessageBox.warning(self, "Could not read file", f"{result.path}\n\n{exc}")
            return

        self.meta_text.setPlainText(meta)
        self.populate_archive_table(archive_rows)
        self.populate_fit_table(fit_rows)

        for info in infos:
            shape = "scalar" if info.shape == () else "x".join(map(str, info.shape))
            label = f"{info.key}    [{shape}, {info.dtype}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, info)
            self.dataset_list.addItem(item)
            self.x_combo.addItem(info.key, info)
            self.current_infos[info.label] = info

        self.show_ndscan_plot(result)
        self.auto_select_monitor_dataset()
        self.loading_file = False
        self.dataset_selection_changed()

    def close_ndscan_plot(self) -> None:
        if self.ndscan_widget is not None:
            self.plot_stack.removeWidget(self.ndscan_widget)
            self.ndscan_widget.setParent(None)
            self.ndscan_widget.deleteLater()
            self.ndscan_widget = None
        if self.ndscan_h5_file is not None:
            self.ndscan_h5_file.close()
            self.ndscan_h5_file = None

    def show_matplotlib_plot(self) -> None:
        self.plot_stack.setCurrentWidget(self.canvas)

    def show_ndscan_plot(self, result: ResultFile) -> None:
        self.close_ndscan_plot()
        try:
            widget, h5_file = create_ndscan_plot_widget(result.path)
        except NdscanPlotError as exc:
            self.show_matplotlib_plot()
            if "Could not import ndscan" in str(exc):
                QMessageBox.warning(self, "Could not load ndscan", str(exc))
            return
        except Exception as exc:
            self.show_matplotlib_plot()
            QMessageBox.warning(self, "Could not show ndscan plot", str(exc))
            return

        self.ndscan_widget = widget
        self.ndscan_h5_file = h5_file
        self.plot_stack.addWidget(widget)
        self.plot_stack.setCurrentWidget(widget)

    def show_current_ndscan_plot(self) -> None:
        if self.current_file is None:
            return
        self.show_ndscan_plot(self.current_file)

    def describe_file(self, h5: h5py.File, result: ResultFile) -> str:
        lines = [
            f"Path: {result.path}",
            f"RID: {result.rid or ''}",
            f"Class: {result.class_name}",
            f"Experiment file: {result.file_name}",
            f"Start: {format_time(result.start_time)}",
            f"Run time: {'' if result.run_time is None else result.run_time}",
        ]
        if "expid" in h5:
            expid = parse_expid(h5["expid"][()])
            args = expid.get("arguments", {})
            if args:
                lines.append("\nArguments:")
                for key, value in sorted(args.items()):
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def populate_archive_table(self, rows: list[tuple[str, str, str]]) -> None:
        self.archive_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                self.archive_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.archive_table.resizeColumnsToContents()

    def collect_fit_rows(self, h5: h5py.File) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for root in ("datasets", "archive"):
            if root not in h5:
                continue

            def visit(name: str, obj: h5py.Dataset) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                lower = name.lower()
                if "fit" not in lower and "err" not in lower and "centre" not in lower:
                    return
                if is_numeric_dataset(obj) or obj.shape == ():
                    rows.append((root, name, summarise_value(obj[()])))

            h5[root].visititems(visit)
        return rows

    def populate_fit_table(self, rows: list[tuple[str, str, str]]) -> None:
        self.fit_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                self.fit_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.fit_table.resizeColumnsToContents()

    def dataset_selection_changed(self) -> None:
        infos = self.selected_infos()
        if len(infos) == 1:
            self.populate_raw_table(infos[0])
            if self.x_combo.currentData() is None:
                self.auto_select_x_for(infos[0])
            if not self.loading_file:
                self.plot_selected()
        elif len(infos) > 1:
            self.clear_raw_table(f"{len(infos)} datasets selected. Select one dataset to inspect raw values.")
        else:
            self.clear_raw_table()

    def clear_raw_table(self, message: str = "Select a dataset to inspect raw values.") -> None:
        self.raw_summary.setText(message)
        self.raw_table.clear()
        self.raw_table.setRowCount(0)
        self.raw_table.setColumnCount(0)

    def populate_raw_table(self, info: DatasetInfo) -> None:
        if self.current_file is None:
            return
        try:
            data = np.asarray(read_dataset(self.current_file.path, info)).squeeze()
        except Exception as exc:
            self.clear_raw_table(f"Could not read {info.label}: {exc}")
            return

        self.raw_summary.setText(
            f"{info.label} | shape={data.shape or 'scalar'} | dtype={data.dtype}"
        )
        self.raw_table.clear()

        max_rows = 1000
        max_cols = 20
        if data.ndim == 0:
            self.raw_table.setRowCount(1)
            self.raw_table.setColumnCount(1)
            self.raw_table.setHorizontalHeaderLabels(["value"])
            self.raw_table.setItem(0, 0, QTableWidgetItem(str(_decode(data.item()))))
        elif data.ndim == 1:
            rows = min(data.size, max_rows)
            self.raw_table.setRowCount(rows)
            self.raw_table.setColumnCount(2)
            self.raw_table.setHorizontalHeaderLabels(["index", "value"])
            for row in range(rows):
                self.raw_table.setItem(row, 0, QTableWidgetItem(str(row)))
                self.raw_table.setItem(row, 1, QTableWidgetItem(str(_decode(data[row]))))
            if data.size > rows:
                self.raw_summary.setText(self.raw_summary.text() + f" | showing first {rows} values")
        elif data.ndim == 2:
            rows = min(data.shape[0], max_rows)
            cols = min(data.shape[1], max_cols)
            self.raw_table.setRowCount(rows)
            self.raw_table.setColumnCount(cols)
            self.raw_table.setHorizontalHeaderLabels([str(col) for col in range(cols)])
            self.raw_table.setVerticalHeaderLabels([str(row) for row in range(rows)])
            for row in range(rows):
                for col in range(cols):
                    self.raw_table.setItem(row, col, QTableWidgetItem(str(_decode(data[row, col]))))
            if data.shape[0] > rows or data.shape[1] > cols:
                self.raw_summary.setText(
                    self.raw_summary.text()
                    + f" | showing {rows}x{cols} of {data.shape[0]}x{data.shape[1]}"
                )
        else:
            flat = data.ravel()
            rows = min(flat.size, max_rows)
            self.raw_table.setRowCount(rows)
            self.raw_table.setColumnCount(2)
            self.raw_table.setHorizontalHeaderLabels(["flat index", "value"])
            for row in range(rows):
                self.raw_table.setItem(row, 0, QTableWidgetItem(str(row)))
                self.raw_table.setItem(row, 1, QTableWidgetItem(str(_decode(flat[row]))))
            self.raw_summary.setText(
                self.raw_summary.text() + f" | flattened, showing first {rows} values"
            )
        self.raw_table.resizeColumnsToContents()

    def export_raw_csv(self) -> None:
        infos = self.selected_infos()
        if self.current_file is None or len(infos) != 1:
            QMessageBox.information(self, "Select one dataset", "Select exactly one dataset to export.")
            return
        info = infos[0]
        try:
            data = np.asarray(read_dataset(self.current_file.path, info)).squeeze()
        except Exception as exc:
            QMessageBox.warning(self, "Could not read dataset", str(exc))
            return

        rid = "unknown" if self.current_file.rid is None else str(self.current_file.rid)
        safe_name = info.label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        default_name = f"{rid}_{safe_name}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export raw dataset", default_name, "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            self.write_csv(Path(path), data)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def write_csv(self, path: Path, data: np.ndarray) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            if data.ndim == 0:
                writer.writerow(["value"])
                writer.writerow([_decode(data.item())])
            elif data.ndim == 1:
                writer.writerow(["index", "value"])
                for idx, value in enumerate(data):
                    writer.writerow([idx, _decode(value)])
            elif data.ndim == 2:
                writer.writerow(["row"] + [str(col) for col in range(data.shape[1])])
                for row_idx, row in enumerate(data):
                    writer.writerow([row_idx] + [_decode(value) for value in row])
            else:
                writer.writerow([f"dim_{idx}" for idx in range(data.ndim)] + ["value"])
                for index in np.ndindex(data.shape):
                    writer.writerow(list(index) + [_decode(data[index])])

    def auto_select_monitor_dataset(self) -> None:
        preferred = ("data/sr/fluo", "data.ca.fluo", "data/sr/fluo", "data.ca.fluo")
        for row in range(self.dataset_list.count()):
            item = self.dataset_list.item(row)
            info = item.data(Qt.UserRole)
            norm = info.key.replace(".", "/").lower()
            if any(p.replace(".", "/") in norm for p in preferred):
                item.setSelected(True)
                self.dataset_list.scrollToItem(item)
                return

    def selected_infos(self) -> list[DatasetInfo]:
        infos = []
        for item in self.dataset_list.selectedItems():
            info = item.data(Qt.UserRole)
            if isinstance(info, DatasetInfo):
                infos.append(info)
        return infos

    def selected_x(self) -> DatasetInfo | None:
        data = self.x_combo.currentData()
        return data if isinstance(data, DatasetInfo) else None

    def auto_select_x_for(self, info: DatasetInfo) -> None:
        x_info = self.find_auto_x_info(info)
        if x_info is None:
            return
        for idx in range(self.x_combo.count()):
            candidate = self.x_combo.itemData(idx)
            if isinstance(candidate, DatasetInfo) and candidate.label == x_info.label:
                self.x_combo.setCurrentIndex(idx)
                return

    def find_auto_x_info(self, info: DatasetInfo) -> DatasetInfo | None:
        if self.current_file is None or info.shape == ():
            return None
        target_len = info.shape[-1]
        candidates = []
        info_prefix = info.key.rsplit("/", 1)[0] if "/" in info.key else ""
        for candidate in self.current_infos.values():
            if candidate.label == info.label or candidate.shape == ():
                continue
            if len(candidate.shape) != 1 or candidate.shape[0] != target_len:
                continue
            key = candidate.key.lower()
            name = key.rsplit("/", 1)[-1]
            if any(part in name for part in ("fit", "err", "error", "sigma")):
                continue

            score = 0
            cand_prefix = candidate.key.rsplit("/", 1)[0] if "/" in candidate.key else ""
            if cand_prefix == info_prefix:
                score += 5
            if name in ("x", "xs", "axis", "points"):
                score += 10
            if any(part in name for part in ("freq", "frequency", "time", "phase", "detuning", "voltage", "power")):
                score += 7
            if any(part in name for part in ("scan", "axis", "point")):
                score += 3
            candidates.append((score, candidate))

        candidates = [item for item in candidates if item[0] > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1].label.lower()))
        return candidates[0][1]

    def plot_selected(self) -> None:
        if self.current_file is None:
            return
        infos = self.selected_infos()
        if not infos:
            return
        self.show_matplotlib_plot()
        self.canvas.clear()
        ax = self.canvas.ax
        last_x_label = "Index"

        for info in infos:
            y = np.asarray(read_dataset(self.current_file.path, info)).squeeze()
            label = info.label
            x_info = self.selected_x() or self.find_auto_x_info(info)
            x_values = None
            if x_info is not None:
                x_values = np.ravel(read_dataset(self.current_file.path, x_info))
            if y.ndim == 0:
                ax.scatter([0], [float(y)], label=label)
            elif y.ndim == 1:
                x = x_values if x_values is not None and len(x_values) == len(y) else np.arange(len(y))
                ax.plot(x, y, marker="o", label=label)
                if x_info is not None and len(x) == len(y):
                    last_x_label = x_info.label
            elif y.ndim == 2:
                extent = None
                if x_values is not None and len(x_values) == y.shape[1]:
                    extent = [float(np.min(x_values)), float(np.max(x_values)), -0.5, y.shape[0] - 0.5]
                    last_x_label = x_info.label if x_info is not None else "Index"
                im = ax.imshow(y, aspect="auto", origin="lower", extent=extent)
                self.canvas.figure.colorbar(im, ax=ax, label=label)
            else:
                ax.plot(np.ravel(y), marker=".", linestyle="", label=f"{label} flattened")
            if self.auto_fit_curves.isChecked():
                self.overlay_fit_for(info, ax)

        ax.set_title(self.current_file.class_name or self.current_file.path.name)
        ax.set_xlabel(last_x_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.canvas.draw()

    def overlay_fit_for(self, info: DatasetInfo, ax) -> None:
        if self.current_file is None:
            return
        candidates = self.fit_curve_candidates(info)
        for x_label, y_label in candidates:
            if x_label not in self.current_infos or y_label not in self.current_infos:
                continue
            x = np.ravel(read_dataset(self.current_file.path, self.current_infos[x_label]))
            y = np.ravel(read_dataset(self.current_file.path, self.current_infos[y_label]))
            if len(x) == len(y) and len(x):
                ax.plot(x, y, "-", linewidth=2, label=f"fit: {y_label}")
                return

    def fit_curve_candidates(self, info: DatasetInfo) -> list[tuple[str, str]]:
        root = info.root
        key = info.key
        base = key.rsplit("/", 1)[-1]
        prefix = key[: -len(base)]
        pairs = [
            (prefix + "fit_x", prefix + "fit_y"),
            (prefix + "fit_xs", prefix + "fit_ys"),
            (prefix + "x_fit", prefix + "y_fit"),
            (prefix + base + "_fit_x", prefix + base + "_fit_y"),
            (prefix + "fit_freqs", prefix + "fit_counts"),
            (prefix + "fit_powers", prefix + "fit_ratios"),
            (prefix + "ion_phase_fit_x", prefix + "ion_phase_fit_y"),
        ]
        labelled = []
        for x, y in pairs:
            labelled.append((f"{root}/{x}", f"{root}/{y}"))
        return labelled

    def plot_history(self) -> None:
        infos = self.selected_infos()
        if not infos:
            return
        self.show_matplotlib_plot()
        self.canvas.clear()
        ax = self.canvas.ax
        plotted = False
        visible_files = [
            self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())
        ]
        visible_files = [f for f in visible_files if isinstance(f, ResultFile)]
        visible_files.sort(key=lambda f: (f.start_time is None, f.start_time or 0, f.rid or 0))

        for info in infos:
            xs = []
            ys = []
            for result in visible_files:
                try:
                    with h5py.File(result.path, "r") as h5:
                        if info.root not in h5 or info.key not in h5[info.root]:
                            continue
                        value = np.asarray(h5[info.root][info.key][()]).squeeze()
                        if value.ndim != 0:
                            continue
                        xs.append(result.start_time if result.start_time is not None else result.rid)
                        ys.append(float(value))
                except Exception:
                    continue
            if xs:
                ax.plot(xs, ys, marker="o", label=info.label)
                plotted = True

        if not plotted:
            QMessageBox.information(
                self,
                "No scalar history",
                "History plotting works for scalar datasets present across result files.",
            )
            return

        if all(isinstance(x, (float, int)) and x > 1_000_000_000 for x in ax.lines[0].get_xdata()):
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels([format_time(t) for t in ticks], rotation=30, ha="right")
            ax.set_xlabel("Start time")
        else:
            ax.set_xlabel("RID")
        ax.set_title("Dataset History")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.canvas.draw()

    def closeEvent(self, event) -> None:
        self.close_ndscan_plot()
        self.summary_timer.stop()
        self.scan_timer.stop()
        if self.pending_scan_future is not None:
            self.pending_scan_future.cancel()
        self.summary_executor.shutdown(wait=False, cancel_futures=True)
        self.scan_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = NightlyMonitorGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
