"""Small bridge for embedding ndscan.show plots in the nightly data GUI."""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import h5py

NDSCAN_AXIS_FONT_SIZE = 12
NDSCAN_LABEL_FONT_SIZE = 13
NDSCAN_BACKGROUND_COLOR = "#000000"
NDSCAN_FOREGROUND_COLOR = "#f0f0f0"
NDSCAN_GRID_ALPHA = 0.2


class NdscanPlotError(RuntimeError):
    pass


def _ensure_ndscan_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    package_roots = [
        root / "vendor",
        root / "Oxford code" / "artiq-oitg" / ".venv" / "Lib" / "site-packages",
    ]
    for package_root in reversed(package_roots):
        if package_root.exists():
            package_root_text = str(package_root)
            if package_root_text not in sys.path:
                sys.path.insert(0, package_root_text)


def create_ndscan_plot_widget(path: Path):
    """Return ``(widget, h5_file)`` for an ndscan HDF5 result file.

    The caller must keep ``h5_file`` alive for as long as the returned Qt widget is
    displayed, because the ndscan models read directly from the open HDF5 datasets.
    """
    _ensure_ndscan_on_path()

    try:
        import pyqtgraph
        from ndscan._qt import QtCore, QtGui, QtWidgets
        from ndscan.plots.container_widgets import PlotAreaTabWidget, PlotAreaWidget
        from ndscan.plots.model import Context
        from ndscan.plots.model.hdf5 import HDF5Root
        from ndscan.results.tools import find_ndscan_roots, get_source_id
        from ndscan.utils import shorten_to_unambiguous_suffixes, strip_suffix
    except Exception as exc:
        raise NdscanPlotError(f"Could not import ndscan plotting tools: {exc}") from exc

    pyqtgraph.setConfigOptions(
        background=NDSCAN_BACKGROUND_COLOR,
        foreground=NDSCAN_FOREGROUND_COLOR,
    )

    h5_file = h5py.File(path, "r")
    try:
        try:
            datasets = h5_file["datasets"]
        except KeyError as exc:
            raise NdscanPlotError("No ARTIQ datasets group found.") from exc

        prefixes = find_ndscan_roots(datasets)
        if not prefixes:
            raise NdscanPlotError("No ndscan result datasets found.")

        title = path.name
        context = Context()
        context.set_source_id(get_source_id(datasets, prefixes))
        roots = [HDF5Root(datasets, prefix, context, title) for prefix in prefixes]

        if len(roots) == 1:
            widget = PlotAreaWidget(roots[0], context)
        else:
            label_map = shorten_to_unambiguous_suffixes(
                prefixes, lambda fqn, n: ".".join(fqn.split(".")[-(n + 1) :])
            )
            widget = PlotAreaTabWidget(
                OrderedDict(
                    zip((strip_suffix(label_map[prefix], ".") for prefix in prefixes), roots)
                ),
                context,
            )
        apply_ndscan_plot_style(widget, pyqtgraph, QtCore, QtGui, QtWidgets)
        return widget, h5_file
    except Exception:
        h5_file.close()
        raise


def apply_ndscan_plot_style(widget, pyqtgraph, QtCore, QtGui, QtWidgets) -> None:
    axis_font = QtGui.QFont()
    axis_font.setPointSize(NDSCAN_AXIS_FONT_SIZE)
    label_font = QtGui.QFont()
    label_font.setPointSize(NDSCAN_LABEL_FONT_SIZE)

    def style_axis(axis) -> None:
        axis.setTickFont(axis_font)
        axis.setPen(pyqtgraph.mkPen(NDSCAN_FOREGROUND_COLOR))
        axis.setTextPen(pyqtgraph.mkPen(NDSCAN_FOREGROUND_COLOR))
        if hasattr(axis, "label"):
            axis.label.setFont(label_font)
            axis.label.setDefaultTextColor(QtGui.QColor(NDSCAN_FOREGROUND_COLOR))

    def style_now() -> None:
        try:
            graphics_widgets = widget.findChildren(pyqtgraph.GraphicsLayoutWidget)
            labels = widget.findChildren(QtWidgets.QLabel)
        except RuntimeError:
            return

        for graphics_widget in graphics_widgets:
            try:
                graphics_widget.setBackground(NDSCAN_BACKGROUND_COLOR)
                if hasattr(graphics_widget, "panes"):
                    for pane in graphics_widget.panes:
                        pane.setTitle(pane.titleLabel.text, color=NDSCAN_FOREGROUND_COLOR)
                        pane.getViewBox().setBackgroundColor(NDSCAN_BACKGROUND_COLOR)
                        pane.showGrid(x=True, y=True, alpha=NDSCAN_GRID_ALPHA)
                        for axis_name in ("left", "right", "top", "bottom"):
                            style_axis(pane.getAxis(axis_name))
                        for axis in getattr(pane, "_additional_right_axes", []):
                            style_axis(axis)
            except RuntimeError:
                continue

        for label in labels:
            try:
                label.setStyleSheet(
                    f"color: {NDSCAN_FOREGROUND_COLOR}; "
                    f"background: {NDSCAN_BACKGROUND_COLOR};"
                )
            except RuntimeError:
                continue

    style_now()
    QtCore.QTimer.singleShot(0, style_now)
    QtCore.QTimer.singleShot(250, style_now)
