from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from app.config import Config
from app.ui.converter_tab import ConverterTab
from app.ui.depth_tab import DepthTab
from app.ui.pipeline_tab import PipelineTab
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
        upscale = UpscaleTab(cfg)
        depth = DepthTab(cfg)
        converter = ConverterTab(cfg)
        tabs.addTab(upscale, "1 · Upscale")
        tabs.addTab(depth, "2 · Depth")
        tabs.addTab(converter, "3 · Converter (SBS)")
        tabs.addTab(PipelineTab(cfg, upscale, depth, converter), "▶ Run all")
        self.setCentralWidget(tabs)
        self.resize(860, 820)

        self._device_label = QLabel("Detecting hardware…")
        self.statusBar().addPermanentWidget(self._device_label)
        self._probe = _DeviceProbe()
        self._probe.result.connect(self._device_label.setText)
        self._probe.start()
