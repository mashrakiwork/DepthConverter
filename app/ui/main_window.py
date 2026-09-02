from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QLabel, QMainWindow, QTabWidget, QWidget,
)

from app.config import Config
from app.ui.converter_tab import ConverterTab
from app.ui.depth_tab import DepthTab
from app.ui.icon import app_icon
from app.ui.pipeline_tab import PipelineTab
from app.ui.theme import DEFAULT_SCALE, SCALES, apply_scale
from app.ui.upscale_tab import UpscaleTab


class _DeviceProbe(QThread):
    """Imports torch off the UI thread (it takes seconds) to show GPU status."""

    result = Signal(str)

    def run(self):
        try:
            from app.core.hardware import device_summary

            self.result.emit(device_summary())
        except Exception as exc:  # noqa: BLE001
            self.result.emit(f"Hardware probe failed: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthConverter - local 2D to 3D VR")
        self.setWindowIcon(app_icon())
        cfg = Config()
        self.cfg = cfg
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        upscale = UpscaleTab(cfg)
        depth = DepthTab(cfg)
        converter = ConverterTab(cfg)
        tabs.addTab(upscale, "1 · Upscale")
        tabs.addTab(depth, "2 · Depth")
        tabs.addTab(converter, "3 · Converter (SBS)")
        tabs.addTab(PipelineTab(cfg, upscale, depth, converter), "▶ Run all")
        self.setCentralWidget(tabs)

        # Every tab scrolls, so the window can go as small as the user wants.
        self.setMinimumSize(560, 400)
        self._restore_geometry()

        self.statusBar().addPermanentWidget(self._scale_box())
        self._device_label = QLabel("Detecting hardware…")
        self.statusBar().addPermanentWidget(self._device_label)
        self._probe = _DeviceProbe()
        self._probe.result.connect(self._device_label.setText)
        self._probe.start()

    def _scale_box(self) -> QWidget:
        """Status-bar zoom for the whole UI - the fix for 'the text is tiny'."""
        box = QComboBox()
        box.setToolTip("Size of all text and controls. Raise it if the fields "
                       "look small on your monitor; the setting is remembered.")
        for pct in SCALES:
            box.addItem(f"{pct}%", pct)
        saved = int(self.cfg.get("ui.scale", DEFAULT_SCALE))
        idx = box.findData(saved)
        box.setCurrentIndex(idx if idx >= 0 else box.findData(DEFAULT_SCALE))
        box.currentIndexChanged.connect(
            lambda _: self._set_scale(box.currentData()))
        return box

    def _set_scale(self, pct: int):
        self.cfg.set("ui.scale", int(pct))
        apply_scale(int(pct))

    def _restore_geometry(self):
        """Open at a comfortable size that always fits the actual screen."""
        saved = self.cfg.get("ui.window_size")
        width, height = (saved if isinstance(saved, list) and len(saved) == 2
                         else (980, 900))
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(int(width), avail.width() - 40)
            height = min(int(height), avail.height() - 60)
        self.resize(max(int(width), 560), max(int(height), 400))

    def closeEvent(self, event):
        self.cfg.set("ui.window_size", [self.width(), self.height()])
        super().closeEvent(event)
