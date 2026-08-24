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
from PyQt5.QtCore import QDate, QPointF, QSignalBlocker, QTimer, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QTextCharFormat, QTextLayout
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
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
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
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
COMPARE_TIME_ROLE = Qt.UserRole + 1


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


def argument_rows_from_expid(expid: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return the same display rows used by the parameter panel and comparison."""
    arguments = expid.get("arguments", {})
    if not isinstance(arguments, dict):
        return []

    rows: list[tuple[str, str, str]] = []
    for name, stored_value in arguments.items():
        if name == "ndscan_params":
            continue
        value = stored_value
        note = ""
        if isinstance(stored_value, dict):
            if "override" in stored_value and stored_value["override"] is not None:
                value = stored_value["override"]
                note = " (override)"
            elif "default" in stored_value:
                value = stored_value["default"]
                note = " (default)"
        rows.append((str(name), f"{summarise_value(value)}{note}", str(name)))

    try:
        vendor_root = Path(__file__).resolve().parent / "vendor"
        vendor_text = str(vendor_root)
        if vendor_root.exists() and vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)
        from ndscan.results.arguments import extract_param_schema, format_numeric, format_scan_range
        params = extract_param_schema(arguments)
    except Exception:
        params = None

    if params:
        schemata = params.get("schemata", {})
        overrides = params.get("overrides", {})
        axes_by_fqn: dict[str, list[dict[str, Any]]] = {}
        for axis in params.get("scan", {}).get("axes", []):
            axes_by_fqn.setdefault(str(axis.get("fqn", "")), []).append(axis)

        def format_param_value(value: Any, schema: dict[str, Any]) -> str:
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    pass
            try:
                return format_numeric(value, schema.get("spec", {}))
            except Exception:
                return summarise_value(value)

        for fqn, schema in schemata.items():
            fqn = str(fqn)
            display_name = str(schema.get("description") or fqn)
            identity = f"{display_name} {fqn}"
            if "default" in schema:
                value = format_param_value(schema["default"], schema)
                rows.append((display_name, f"{value} (default)", identity))
            for override in overrides.get(fqn, []):
                path = str(override.get("path") or "*")
                value = format_param_value(override.get("value"), schema)
                rows.append((f"{display_name} @ {path}", f"{value} (override)", f"{identity} {path}"))
            for axis in axes_by_fqn.get(fqn, []):
                path = str(axis.get("path") or "*")
                try:
                    value = format_scan_range(str(axis.get("type", "")), axis.get("range", {}), schema)
                except Exception:
                    value = summarise_value(axis.get("range", {}))
                rows.append((f"{display_name} @ {path}", f"{value} (scan)", f"{identity} {path}"))
    elif "ndscan_params" in arguments:
        rows.append(("ndscan_params", summarise_value(arguments["ndscan_params"]), "ndscan_params"))

    return sorted(rows, key=lambda row: (row[0].casefold(), row[1].casefold()))


def parameter_rows_for_comparison(rows: list[tuple[str, str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, value, identity in rows:
        kind_match = re.search(r"\((default|override|scan)\)$", value)
        kind = kind_match.group(1) if kind_match else ""
        label = f"{name} [{kind}]" if kind else name
        if label in values:
            label = f"{label} — {identity}"
        suffix = 2
        unique_label = label
        while unique_label in values:
            unique_label = f"{label} #{suffix}"
            suffix += 1
        values[unique_label] = value
    return values


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


class SearchHighlightDelegate(QStyledItemDelegate):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.terms: list[str] = []

    def set_query(self, query: str) -> None:
        self.terms = query.strip().casefold().split()
        parent = self.parent()
        if isinstance(parent, QAbstractItemView):
            parent.viewport().update()

    def paint(self, painter, option, index) -> None:
        text = str(index.data(Qt.DisplayRole) or "")
        if not text or not self.terms:
            super().paint(painter, option, index)
            return
        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style = style_option.widget.style() if style_option.widget else QApplication.style()
        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, style_option, style_option.widget)
        display_text = style_option.fontMetrics.elidedText(text, Qt.ElideRight, max(0, text_rect.width()))
        folded_text = display_text.casefold()
        matches: list[tuple[int, int]] = []
        for term in self.terms:
            start = 0
            while term:
                found = folded_text.find(term, start)
                if found < 0:
                    break
                matches.append((found, len(term)))
                start = found + len(term)
        if not matches:
            super().paint(painter, option, index)
            return
        style_option.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, style_option, painter, style_option.widget)
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor(255, 225, 80))
        highlight_format.setForeground(QColor(30, 30, 30))
        ranges = []
        for start, length in matches:
            text_range = QTextLayout.FormatRange()
            text_range.start = start
            text_range.length = length
            text_range.format = highlight_format
            ranges.append(text_range)
        layout = QTextLayout(display_text, style_option.font)
        layout.setFormats(ranges)
        layout.beginLayout()
        line = layout.createLine()
        line.setLineWidth(max(0, text_rect.width()))
        layout.endLayout()
        color_role = QPalette.HighlightedText if style_option.state & QStyle.State_Selected else QPalette.Text
        painter.save()
        painter.setClipRect(text_rect)
        painter.setPen(style_option.palette.color(color_role))
        layout.draw(painter, QPointF(text_rect.x(), text_rect.y() + (text_rect.height() - line.height()) / 2))
        painter.restore()


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
        self.parameter_rows: list[tuple[str, str, str]] = []
        self.parameter_popout: QDialog | None = None
        self.panel_popouts: dict[QWidget, QDialog] = {}
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
        self.archive_filter_edit = QLineEdit()
        self.archive_filter_edit.setPlaceholderText("Search archived dataset names, types, and values...")
        self.archive_table = QTableWidget(0, 3)
        self.archive_table.setHorizontalHeaderLabels(["Name", "Type", "Value"])
        archive_header = self.archive_table.horizontalHeader()
        archive_header.setSectionResizeMode(0, QHeaderView.Stretch)
        archive_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        archive_header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.archive_table.setAlternatingRowColors(True)
        self.archive_table.setItemDelegate(SearchHighlightDelegate(self.archive_table))
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
        self.meta_text.setMaximumHeight(110)
        self.argument_filter_edit = QLineEdit()
        self.argument_filter_edit.setPlaceholderText("Search names, values, paths, types, defaults, overrides...")
        self.argument_filter_edit.setClearButtonEnabled(True)
        self.parameter_popout_button = QPushButton()
        self.parameter_popout_button.setEnabled(False)
        zoom_icon = QIcon.fromTheme("zoom-in")
        if zoom_icon.isNull():
            zoom_icon = self.style().standardIcon(QStyle.SP_TitleBarMaxButton)
        self.parameter_popout_button.setIcon(zoom_icon)
        self.parameter_popout_button.setAccessibleName("Zoom in parameters")
        self.parameter_popout_button.setToolTip("Zoom in: open all parameters and full identities in a larger window.")
        self.argument_count_label = QLabel("0 parameters")
        self.argument_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.argument_table = QTableWidget(0, 2)
        self.argument_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        argument_header = self.argument_table.horizontalHeader()
        argument_header.setStretchLastSection(True)
        argument_header.setMinimumSectionSize(0)
        argument_header.setSectionResizeMode(QHeaderView.Interactive)
        self.argument_table.setAlternatingRowColors(True)
        self.argument_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.argument_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.argument_table.setSortingEnabled(True)
        self.argument_table.setWordWrap(False)
        self.argument_table.verticalHeader().setVisible(False)
        self.argument_table.setItemDelegate(SearchHighlightDelegate(self.argument_table))
        self.fit_table = QTableWidget(0, 3)
        self.fit_table.setHorizontalHeaderLabels(["Source", "Name", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.raw_summary = QLabel("Select a dataset to inspect raw values.")
        self.raw_table = QTableWidget(0, 0)
        self.raw_table.setAlternatingRowColors(True)
        self.raw_table.setTextElideMode(Qt.ElideRight)
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
        self.log_errors_only = QCheckBox("Errors only")
        self.log_errors_only.setChecked(True)
        self.log_status = QLabel()
        self.log_status.setMinimumWidth(220)
        self.log_status.setStyleSheet("color: #9a5a00;")
        self.log_file_list = QListWidget()
        self.log_text = QTreeWidget()
        self.log_text.setItemDelegate(SearchHighlightDelegate(self.log_text))
        self.log_text.setHeaderHidden(True)
        self.log_text.setUniformRowHeights(True)
        self.log_text.setRootIsDecorated(True)
        self.log_text.setAlternatingRowColors(True)
        log_font = QFont("Consolas")
        log_font.setStyleHint(QFont.Monospace)
        self.log_text.setFont(log_font)

        self.compare_left = QComboBox()
        self.compare_right = QComboBox()
        for combo in (self.compare_left, self.compare_right):
            combo.setEditable(True)
            combo.setMaximumWidth(720)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.lineEdit().setPlaceholderText("Select a result or type RID, date/RID, or file path")
        self.compare_left_time = QLabel()
        self.compare_right_time = QLabel()
        for label in (self.compare_left_time, self.compare_right_time):
            label.setMinimumWidth(62)
            label.setStyleSheet("color: #888888;")
        self.compare_left_browse = QPushButton("From Folder…")
        self.compare_right_browse = QPushButton("From Folder…")
        self.compare_button = QPushButton("Compare")
        self.compare_button.setMinimumSize(180, 42)
        compare_font = self.compare_button.font()
        compare_font.setBold(True)
        self.compare_button.setFont(compare_font)
        self.compare_status = QLabel("Select two different result files, then click Compare.")
        self.compare_filter_edit = QLineEdit()
        self.compare_filter_edit.setPlaceholderText("Search settings, file values, and differences...")
        self.compare_table = QTableWidget(0, 4)
        self.compare_table.setHorizontalHeaderLabels(["Setting", "First file", "Second file", "Difference"])
        compare_header = self.compare_table.horizontalHeader()
        compare_header.setStretchLastSection(False)
        for column in range(3):
            compare_header.setSectionResizeMode(column, QHeaderView.Stretch)
        compare_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compare_table.setItemDelegate(SearchHighlightDelegate(self.compare_table))

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

    def make_zoom_button(self, title: str, panel: QWidget) -> QPushButton:
        button = QPushButton()
        icon = QIcon.fromTheme("zoom-in")
        if icon.isNull():
            icon = self.style().standardIcon(QStyle.SP_TitleBarMaxButton)
        button.setIcon(icon)
        button.setAccessibleName(f"Zoom in {title}")
        button.setToolTip(f"Zoom in: open {title} in a larger window.")
        button.clicked.connect(lambda _checked=False: self.toggle_panel_popout(panel, title))
        return button

    def toggle_panel_popout(self, panel: QWidget, title: str) -> None:
        existing = self.panel_popouts.get(panel)
        if existing is not None:
            existing.close()
            return
        splitter = panel.parentWidget()
        if not isinstance(splitter, QSplitter):
            return
        panel_index = splitter.indexOf(panel)
        splitter_sizes = splitter.sizes()
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle(f"{title} — Zoomed")
        dialog.resize(1100, 760)
        layout = QVBoxLayout(dialog)
        layout.addWidget(panel)
        self.panel_popouts[panel] = dialog
        dialog.finished.connect(lambda _result: self.restore_zoomed_panel(panel, dialog, splitter, panel_index, splitter_sizes))
        dialog.show()

    def restore_zoomed_panel(self, panel: QWidget, dialog: QDialog, splitter: QSplitter, panel_index: int, sizes: list[int]) -> None:
        if panel.parentWidget() is dialog:
            if dialog.layout() is not None:
                dialog.layout().removeWidget(panel)
            splitter.insertWidget(panel_index, panel)
            splitter.setSizes(sizes)
        if self.panel_popouts.get(panel) is dialog:
            del self.panel_popouts[panel]

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
        filter_row.addWidget(self.make_zoom_button("Result Files", files_box))
        files_layout = QVBoxLayout(files_box)
        files_layout.addLayout(filter_row)
        files_layout.addWidget(self.file_list)

        data_box = QGroupBox("Result Datasets (Plot)")
        data_zoom_button = self.make_zoom_button("Result Datasets", data_box)
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
        button_row.addWidget(data_zoom_button)
        data_layout.addLayout(button_row)
        data_layout.addWidget(self.auto_fit_curves)

        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(files_box)
        left_split.addWidget(data_box)

        archive_box = QGroupBox("Archive Parameters")
        archive_layout = QVBoxLayout(archive_box)
        archive_search_row = QHBoxLayout()
        archive_search_row.addWidget(self.archive_filter_edit, 1)
        archive_search_row.addWidget(self.make_zoom_button("Archive Parameters", archive_box))
        archive_layout.addLayout(archive_search_row)
        archive_layout.addWidget(self.archive_table)
        left_split.addWidget(archive_box)
        left_split.setSizes([330, 260, 220])

        details_split = QSplitter(Qt.Vertical)

        details_box = QGroupBox("Experiment Details")
        details_layout = QVBoxLayout(details_box)
        details_layout.addWidget(self.meta_text)
        argument_filter_row = QHBoxLayout()
        argument_filter_row.addWidget(QLabel("Arguments"))
        argument_filter_row.addWidget(self.argument_filter_edit, 1)
        argument_filter_row.addWidget(self.argument_count_label)
        argument_filter_row.addWidget(self.parameter_popout_button)
        details_layout.addLayout(argument_filter_row)
        details_layout.addWidget(self.argument_table, 1)
        details_split.addWidget(details_box)

        fit_box = QGroupBox()
        fit_box.setAccessibleName("Fit-Related Results")
        fit_layout = QVBoxLayout(fit_box)
        fit_header = QHBoxLayout()
        fit_title = QLabel("Fit-Related Results")
        fit_title.setStyleSheet("font-weight: bold;")
        fit_header.addWidget(fit_title)
        fit_header.addStretch(1)
        fit_header.addWidget(self.make_zoom_button("Fit-Related Results", fit_box))
        fit_layout.addLayout(fit_header)
        fit_layout.addWidget(self.fit_table)
        details_split.addWidget(fit_box)

        raw_box = QGroupBox("Raw Data")
        raw_layout = QVBoxLayout(raw_box)
        raw_summary_row = QHBoxLayout()
        raw_summary_row.addWidget(self.raw_summary, 1)
        raw_summary_row.addWidget(self.make_zoom_button("Raw Data", raw_box))
        raw_layout.addLayout(raw_summary_row)
        raw_layout.addWidget(self.raw_table)
        raw_layout.addWidget(self.export_raw_button)
        details_split.addWidget(raw_box)
        details_split.setSizes([300, 170, 240])

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left_split)
        plot_box = QGroupBox()
        plot_box.setAccessibleName("Plot")
        plot_layout = QVBoxLayout(plot_box)
        plot_header = QHBoxLayout()
        plot_title = QLabel("Plot")
        plot_title.setStyleSheet("font-weight: bold;")
        plot_header.addWidget(plot_title)
        plot_header.addStretch(1)
        plot_header.addWidget(self.make_zoom_button("Plot", plot_box))
        plot_layout.addLayout(plot_header)
        plot_layout.addWidget(self.plot_stack, 1)
        main_split.addWidget(plot_box)
        main_split.addWidget(details_split)
        main_split.setSizes([340, 600, 360])

        data_tab = QWidget()
        data_layout_root = QVBoxLayout(data_tab)
        data_layout_root.addLayout(top)
        data_layout_root.addWidget(main_split, 1)

        log_tab = self.build_log_tab()
        compare_tab = self.build_compare_tab()

        tabs = QTabWidget()
        tabs.addTab(data_tab, "Data")
        tabs.addTab(log_tab, "Logs")
        tabs.addTab(compare_tab, "Compare Parameters")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(tabs)
        self.setCentralWidget(root)

    def build_compare_tab(self) -> QWidget:
        controls = QGridLayout()
        controls.addWidget(QLabel("First result"), 0, 0)
        controls.addWidget(self.compare_left, 0, 1)
        controls.addWidget(self.compare_left_time, 0, 2)
        controls.addWidget(self.compare_left_browse, 0, 3)
        controls.addWidget(QLabel("Second result"), 1, 0)
        controls.addWidget(self.compare_right, 1, 1)
        controls.addWidget(self.compare_right_time, 1, 2)
        controls.addWidget(self.compare_right_browse, 1, 3)
        controls.addWidget(self.compare_button, 1, 4)
        controls.setColumnStretch(5, 1)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addLayout(controls)
        layout.addWidget(self.compare_status)
        layout.addWidget(self.compare_filter_edit)
        layout.addWidget(self.compare_table, 1)
        return tab

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
        filter_row.addWidget(self.log_errors_only)
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
        self.argument_filter_edit.textChanged.connect(self.filter_arguments)
        self.parameter_popout_button.clicked.connect(self.show_parameter_popout)
        self.archive_filter_edit.textChanged.connect(lambda text: self.filter_table(self.archive_table, text))
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
        self.log_errors_only.stateChanged.connect(lambda _state: self.load_selected_log())
        self.log_time_filter.stateChanged.connect(lambda _state: self.load_selected_log())
        self.log_file_list.currentItemChanged.connect(lambda current, _previous: self.log_file_selected(current))
        self.compare_button.clicked.connect(self.compare_selected_files)
        self.compare_left_browse.clicked.connect(lambda: self.browse_compare_file(self.compare_left, "first"))
        self.compare_right_browse.clicked.connect(lambda: self.browse_compare_file(self.compare_right, "second"))
        self.compare_left.lineEdit().returnPressed.connect(self.compare_selected_files)
        self.compare_right.lineEdit().returnPressed.connect(self.compare_selected_files)
        self.compare_left.currentIndexChanged.connect(lambda _index: self.update_compare_time_labels())
        self.compare_right.currentIndexChanged.connect(lambda _index: self.update_compare_time_labels())
        self.compare_left.lineEdit().textEdited.connect(lambda _text: self.compare_left_time.clear())
        self.compare_right.lineEdit().textEdited.connect(lambda _text: self.compare_right_time.clear())
        self.compare_filter_edit.textChanged.connect(lambda text: self.filter_table(self.compare_table, text))
        QTimer.singleShot(0, self.refresh_logs)

    def update_compare_time_labels(self) -> None:
        for combo, label in ((self.compare_left, self.compare_left_time), (self.compare_right, self.compare_right_time)):
            start_time = combo.currentData(COMPARE_TIME_ROLE)
            if isinstance(start_time, (int, float)):
                label.setText(datetime.fromtimestamp(start_time).strftime("%H:%M:%S"))
            else:
                label.clear()

    def update_compare_files(self) -> None:
        selections = []
        for combo in (self.compare_left, self.compare_right):
            text = combo.currentText()
            index = combo.currentIndex()
            data = combo.itemData(index) if index >= 0 and text == combo.itemText(index) else None
            selections.append((text, data))
        for position, (combo, (text, selected_path)) in enumerate(zip((self.compare_left, self.compare_right), selections)):
            blocker = QSignalBlocker(combo)
            combo.clear()
            for result in self.files:
                combo.addItem(str(result.path), result.path)
                combo.setItemData(combo.count() - 1, result.start_time, COMPARE_TIME_ROLE)
                combo.setItemData(combo.count() - 1, str(result.path), Qt.ToolTipRole)
            if isinstance(selected_path, Path):
                index = combo.findData(selected_path)
                if index >= 0:
                    combo.setCurrentIndex(index)
                elif selected_path.is_file():
                    self.add_compare_path(combo, selected_path)
            elif text:
                combo.setEditText(text)
            elif combo.count():
                combo.setCurrentIndex(min(position, combo.count() - 1))
            del blocker
        self.update_compare_time_labels()

    def add_compare_path(self, combo: QComboBox, path: Path) -> None:
        result = read_file_summary(path)
        combo.addItem(str(path), path)
        index = combo.count() - 1
        combo.setItemData(index, result.start_time, COMPARE_TIME_ROLE)
        combo.setItemData(index, str(path), Qt.ToolTipRole)
        combo.setCurrentIndex(index)

    def browse_compare_file(self, combo: QComboBox, side: str) -> None:
        start_path = Path(self.path_edit.text()).expanduser()
        if not path_is_available(start_path):
            start_path = Path(DEFAULT_RESULTS_DIR)
        filename, _ = QFileDialog.getOpenFileName(self, f"Select {side} result file", str(start_path), "HDF5 result files (*.h5 *.hdf5);;All files (*)")
        if filename:
            self.set_compare_combo_path(combo, Path(filename))

    def set_compare_combo_path(self, combo: QComboBox, path: Path) -> None:
        path = path.resolve()
        index = combo.findData(path)
        if index < 0:
            self.add_compare_path(combo, path)
        else:
            combo.setCurrentIndex(index)
        self.update_compare_time_labels()

    def resolve_compare_path(self, combo: QComboBox) -> Path:
        raw_text = combo.currentText()
        index = combo.currentIndex()
        if index >= 0 and raw_text == combo.itemText(index):
            path = combo.itemData(index)
            if isinstance(path, Path) and path.is_file():
                return path
        text = raw_text.strip()
        if not text:
            raise ValueError("no result was selected or entered")
        entered = Path(text.strip('"')).expanduser()
        current_root = Path(self.path_edit.text()).expanduser()
        candidates = [entered]
        if not entered.is_absolute():
            candidates.extend([current_root / entered, Path(DEFAULT_RESULTS_DIR) / entered])
            if LOG_DATE_RE.fullmatch(current_root.name):
                candidates.append(current_root.parent / entered)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
            if candidate.is_dir():
                files = list(candidate.glob("*.h5")) + list(candidate.glob("*.hdf5"))
                if len(files) == 1:
                    return files[0].resolve()
                if files:
                    raise ValueError(f"folder contains {len(files)} result files; add a RID or choose one file")
        date_match = LOG_DATE_RE.search(text)
        date_text = date_match.group(1) if date_match else ""
        rid_matches = re.findall(r"(?<!\d)\d+(?!\d)", text.replace(date_text, " ") if date_text else text)
        if not rid_matches:
            raise ValueError("entry is not an existing file; enter a RID, date/RID, or complete file path")
        rid = int(rid_matches[-1])
        for result in self.files:
            if result.rid == rid and (not date_text or date_text in result.path.parts):
                return result.path
        roots = [current_root, Path(DEFAULT_RESULTS_DIR)]
        for root in roots:
            if date_text:
                root = root.parent / date_text if LOG_DATE_RE.fullmatch(root.name) else root / date_text
            if not path_is_available(root):
                continue
            for pattern in ("*.h5", "*.hdf5"):
                for path in root.rglob(pattern):
                    if rid_from_path(path) == rid:
                        return path.resolve()
        raise ValueError(f"RID {rid}{' on ' + date_text if date_text else ''} was not found")

    def comparison_values(self, path: Path) -> dict[str, Any]:
        result = read_file_summary(path)
        with h5py.File(path, "r") as h5:
            expid = parse_expid(h5["expid"][()] if "expid" in h5 else "")
        values: dict[str, Any] = {
            "RID": result.rid,
            "Start time": result.start_time,
            "Run time": result.run_time,
            "Class": result.class_name,
            "Experiment file": result.file_name,
        }
        params = parameter_rows_for_comparison(argument_rows_from_expid(expid))
        values.update({f"Parameter: {name}": value for name, value in params.items()})
        return values

    def compare_selected_files(self) -> None:
        self.compare_table.clearContents()
        self.compare_table.setRowCount(0)
        try:
            left = self.resolve_compare_path(self.compare_left)
            right = self.resolve_compare_path(self.compare_right)
        except (OSError, ValueError) as exc:
            self.compare_status.setText(str(exc))
            return
        if left.resolve() == right.resolve():
            self.compare_status.setText("The same result file is selected twice. Choose two different files.")
            return
        self.set_compare_combo_path(self.compare_left, left)
        self.set_compare_combo_path(self.compare_right, right)
        try:
            left_values = self.comparison_values(left)
            right_values = self.comparison_values(right)
        except Exception as exc:
            self.compare_status.setText(f"Could not compare files: {exc}")
            return
        priority = ["RID", "Start time", "Run time", "Class", "Experiment file"]
        keys = sorted(set(left_values) | set(right_values), key=lambda key: (left_values.get(key, "<missing>") == right_values.get(key, "<missing>"), priority.index(key) if key in priority else len(priority), str(key).casefold()))
        self.compare_table.setRowCount(len(keys))
        self.compare_table.setHorizontalHeaderLabels(["Setting", left.name, right.name, "Difference"])
        changed = 0
        for row, key in enumerate(keys):
            left_value = left_values.get(key, "<missing>")
            right_value = right_values.get(key, "<missing>")
            equal = left_value == right_value
            changed += not equal
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                difference = right_value - left_value
            else:
                difference = "same" if equal else "changed"
            display = lambda value: format_time(value) if key == "Start time" else str(value)
            items = [QTableWidgetItem(str(key)), QTableWidgetItem(display(left_value)), QTableWidgetItem(display(right_value)), QTableWidgetItem(str(difference))]
            if not equal:
                items[3].setBackground(QColor(255, 220, 220))
            for column, item in enumerate(items):
                self.compare_table.setItem(row, column, item)
        self.filter_table(self.compare_table, self.compare_filter_edit.text())
        self.compare_status.setText(f"Compared {left.name} with {right.name}: <b>{changed} out of {len(keys)} settings differ.</b>")

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
        delegate = self.log_text.itemDelegate()
        if isinstance(delegate, SearchHighlightDelegate):
            delegate.set_query(self.log_filter_edit.text())
        item = self.log_file_list.currentItem()
        if item is None:
            return
        log_file = item.data(Qt.UserRole)
        if not isinstance(log_file, LogFile):
            return

        filter_terms = self.log_filter_edit.text().strip().casefold().split()
        time_filter_enabled = self.log_time_filter.isChecked()
        errors_only_enabled = self.log_errors_only.isChecked()
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
            if errors_only_enabled and not (self.log_line_level(current_title) == "error" or any(self.log_line_level(child) == "error" for child in current_children)):
                current_title = ""
                current_children = []
                return
            if filter_terms:
                folded_title = current_title.casefold()
                title_matches = all(term in folded_title for term in filter_terms)
                matching_children = [
                    child for child in current_children
                    if all(term in f"{folded_title} {child.casefold()}" for term in filter_terms)
                ]
                combined = " ".join([folded_title, *(child.casefold() for child in current_children)])
                if not all(term in combined for term in filter_terms):
                    current_title = ""
                    current_children = []
                    return
                display_children = current_children if title_matches or not matching_children else matching_children
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
        if filter_terms or time_filter_enabled or errors_only_enabled:
            scopes = []
            if errors_only_enabled:
                scopes.append("error")
            if time_filter_enabled:
                scopes.append("1am-7am")
            if filter_terms:
                scopes.append("text-filtered")
            self.show_log_status(f"Showing {shown:,} of {matched:,} {', '.join(scopes)} line(s) in {log_file.path.name}.")
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
        if self.log_filter_edit.text().strip() and children:
            parent.setExpanded(True)

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
        self.update_compare_files()
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
        if self.parameter_popout is not None:
            self.parameter_popout.close()
        self.loading_file = True
        self.dataset_list.clear()
        self.archive_table.setRowCount(0)
        self.argument_table.setRowCount(0)
        self.parameter_rows = []
        self.parameter_popout_button.setEnabled(False)
        self.update_argument_count()
        self.clear_raw_table()
        self.x_combo.clear()
        self.x_combo.addItem("Index / scalar history")
        self.current_infos = {}
        try:
            with h5py.File(result.path, "r") as h5:
                infos = iter_numeric_datasets(h5, roots=("datasets",))
                archive_rows = iter_archive_rows(h5)
                meta = self.describe_file(h5, result)
                argument_rows = self.collect_argument_rows(h5)
                fit_rows = self.collect_fit_rows(h5)
        except Exception as exc:
            self.loading_file = False
            QMessageBox.warning(self, "Could not read file", f"{result.path}\n\n{exc}")
            return

        self.meta_text.setPlainText(meta)
        self.populate_argument_table(argument_rows)
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
        return "\n".join(lines)

    def collect_argument_rows(self, h5: h5py.File) -> list[tuple[str, str, str]]:
        if "expid" not in h5:
            return []
        return argument_rows_from_expid(parse_expid(h5["expid"][()]))

    def populate_argument_table(self, rows: list[tuple[str, str, str]]) -> None:
        self.parameter_rows = list(rows)
        self.parameter_popout_button.setEnabled(bool(rows))
        self.argument_table.setSortingEnabled(False)
        self.argument_table.setRowCount(len(rows))
        for row_idx, (name, value, identity) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            value_item = QTableWidgetItem(value)
            name_item.setData(Qt.UserRole, identity)
            name_item.setToolTip(identity)
            value_item.setToolTip(value)
            self.argument_table.setItem(row_idx, 0, name_item)
            self.argument_table.setItem(row_idx, 1, value_item)
        self.argument_table.setSortingEnabled(True)
        self.argument_table.sortItems(0, Qt.AscendingOrder)
        available_width = self.argument_table.viewport().width()
        if available_width > 0:
            self.argument_table.setColumnWidth(0, max(120, int(available_width * 0.45)))
        self.filter_arguments()

    def filter_arguments(self, _text: str = "") -> None:
        self.filter_table(self.argument_table, self.argument_filter_edit.text())
        self.update_argument_count()

    @staticmethod
    def filter_table(table: QTableWidget, query: str) -> None:
        terms = query.strip().casefold().split()
        delegate = table.itemDelegate()
        if isinstance(delegate, SearchHighlightDelegate):
            delegate.set_query(query)
        for row in range(table.rowCount()):
            parts = []
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is None:
                    continue
                for role in (Qt.DisplayRole, Qt.ToolTipRole, Qt.UserRole):
                    value = item.data(role)
                    if value is not None:
                        parts.append(str(value))
            searchable = " ".join(parts).casefold()
            table.setRowHidden(row, any(term not in searchable for term in terms))

    def show_parameter_popout(self) -> None:
        if not self.parameter_rows:
            return
        if self.parameter_popout is not None:
            self.parameter_popout.close()
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle("All Experiment Parameters / Settings")
        dialog.resize(1050, 720)
        layout = QVBoxLayout(dialog)
        search = QLineEdit()
        search.setPlaceholderText("Search all names, values, full identities, paths, and setting types...")
        layout.addWidget(search)
        table = QTableWidget(len(self.parameter_rows), 3)
        table.setHorizontalHeaderLabels(["Parameter", "Value", "Full identity / path"])
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setItemDelegate(SearchHighlightDelegate(table))
        for row_idx, row in enumerate(self.parameter_rows):
            for column, text in enumerate(row):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(row_idx, column, item)
        for column in range(3):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        layout.addWidget(table, 1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button, 0, Qt.AlignRight)
        search.textChanged.connect(lambda text: self.filter_table(table, text))
        dialog.finished.connect(lambda _result: self.clear_parameter_popout(dialog))
        self.parameter_popout = dialog
        dialog.show()

    def clear_parameter_popout(self, dialog: QDialog) -> None:
        if self.parameter_popout is dialog:
            self.parameter_popout = None

    def update_argument_count(self) -> None:
        total = self.argument_table.rowCount()
        visible = sum(not self.argument_table.isRowHidden(row) for row in range(total))
        if visible == total:
            label = f"{total} parameter{'s' if total != 1 else ''}"
        else:
            label = f"{visible} of {total}"
        self.argument_count_label.setText(label)

    def populate_archive_table(self, rows: list[tuple[str, str, str]]) -> None:
        self.archive_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.archive_table.setItem(row_idx, col_idx, item)
        self.filter_table(self.archive_table, self.archive_filter_edit.text())

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
        for row in range(self.raw_table.rowCount()):
            for column in range(self.raw_table.columnCount()):
                item = self.raw_table.item(row, column)
                if item is not None:
                    item.setToolTip(item.text())
        self.fit_raw_table_columns()

    def fit_raw_table_columns(self) -> None:
        count = self.raw_table.columnCount()
        if not count:
            return
        header = self.raw_table.horizontalHeader()
        header.setStretchLastSection(False)
        if count == 1:
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            return
        if count == 2:
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            return
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.raw_table.resizeColumnsToContents()
        target = max(44, min(90, self.raw_table.viewport().width() // min(count, 8)))
        for column in range(count):
            header.resizeSection(column, max(44, min(self.raw_table.columnWidth(column), target)))

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
