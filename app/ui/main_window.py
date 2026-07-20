from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from app.config import Config
from app.ui.converter_tab import ConverterTab
from app.ui.depth_tab import DepthTab
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
        cfg = Config()
        tabs = QTabWidget()
        tabs.addTab(UpscaleTab(cfg), "1 · Upscale")
        tabs.addTab(DepthTab(cfg), "2 · Depth")
        tabs.addTab(ConverterTab(cfg), "3 · Converter (SBS)")
        self.setCentralWidget(tabs)
        self.resize(860, 820)

        self._device_label = QLabel("Detecting hardware…")
        self.statusBar().addPermanentWidget(self._device_label)
        self._probe = _DeviceProbe()
        self._probe.result.connect(self._device_label.setText)
        self._probe.start()
