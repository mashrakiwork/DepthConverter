"""Shared UI building blocks: background job worker, path pickers, encoding box."""

import threading
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from app.core.video_io import available_encoders

VIDEO_FILTER = "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mts *.ts);;All files (*)"
MEDIA_FILTER = ("Media (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mts *.ts "
                "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;All files (*)")


class Worker(QThread):
    progress = Signal(float, str)
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, fn, opts: dict):
        super().__init__()
        self._fn = fn
        self._opts = opts
        self.cancel_event = threading.Event()

    def run(self):
        from app.core.pipeline import JobCancelled

        try:
            self._fn(self._opts,
                     progress=lambda p, m: self.progress.emit(p, m),
                     log=lambda m: self.log.emit(str(m)),
                     cancel=self.cancel_event)
            self.done.emit(True, "Done")
        except JobCancelled:
            self.done.emit(False, "Cancelled")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.log.emit(traceback.format_exc())
            self.done.emit(False, f"Error: {exc}")


class PathPicker(QWidget):
    """Line edit + browse button(s); value persisted in config under `key`.

    mode: "dir" for a folder, "any" for file-or-folder.
    """

    def __init__(self, cfg, key: str, mode: str = "dir", placeholder: str = "",
                 file_filter: str = VIDEO_FILTER):
        super().__init__()
        self.cfg = cfg
        self.key = key
        self.file_filter = file_filter
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(cfg.get(key, ""))
        self.edit.setPlaceholderText(placeholder)
        self.edit.editingFinished.connect(self._save)
        row.addWidget(self.edit, 1)
        if mode == "dir":
            btn = QPushButton("Folder…")
            btn.clicked.connect(self._pick_dir)
            row.addWidget(btn)
        else:
            fbtn = QPushButton("File…")
            fbtn.clicked.connect(self._pick_file)
            row.addWidget(fbtn)
            dbtn = QPushButton("Folder…")
            dbtn.clicked.connect(self._pick_dir)
            row.addWidget(dbtn)

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose folder", self.edit.text())
        if path:
            self.edit.setText(path)
            self._save()

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", self.edit.text(),
                                              self.file_filter)
        if path:
            self.edit.setText(path)
            self._save()

    def _save(self):
        self.cfg.set(self.key, self.edit.text().strip())

    def path(self) -> str:
        return self.edit.text().strip()


class EncodingGroup(QGroupBox):
    """Encoder / quality / preset controls, only listing encoders the local
    ffmpeg actually supports."""

    def __init__(self, cfg, prefix: str):
        super().__init__("Output encoding")
        self.cfg = cfg
        self.prefix = prefix
        grid = QGridLayout(self)

        self.encoder = QComboBox()
        for key, label in available_encoders():
            self.encoder.addItem(label, key)
        saved = cfg.get(prefix + "encoder")
        idx = self.encoder.findData(saved)
        if idx >= 0:
            self.encoder.setCurrentIndex(idx)
        self.encoder.currentIndexChanged.connect(
            lambda _: cfg.set(prefix + "encoder", self.encoder.currentData()))

        self.quality = QSpinBox()
        self.quality.setRange(0, 51)
        self.quality.setValue(int(cfg.get(prefix + "quality", 18)))
        self.quality.setToolTip("CRF / CQ - lower means better quality and bigger files. "
                                "18 is visually near-lossless.")
        self.quality.valueChanged.connect(lambda v: cfg.set(prefix + "quality", v))

        self.preset = QComboBox()
        self.preset.addItems(["quality", "balanced", "fast"])
        self.preset.setCurrentText(cfg.get(prefix + "preset", "balanced"))
        self.preset.currentTextChanged.connect(lambda t: cfg.set(prefix + "preset", t))

        grid.addWidget(QLabel("Codec:"), 0, 0)
        grid.addWidget(self.encoder, 0, 1)
        grid.addWidget(QLabel("Quality (CRF/CQ):"), 1, 0)
        grid.addWidget(self.quality, 1, 1)
        grid.addWidget(QLabel("Speed preset:"), 2, 0)
        grid.addWidget(self.preset, 2, 1)
        grid.setColumnStretch(1, 1)

    def opts(self) -> dict:
        return {
            "encoder": self.encoder.currentData(),
            "quality": self.quality.value(),
            "preset": self.preset.currentText(),
        }


class JobTab(QWidget):
    """Base for both tabs: subclasses build their form, then call
    _add_run_controls(layout); build_opts() + job_fn() drive the worker."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.worker: Worker | None = None

    def _add_run_controls(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.start_btn, 1)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.status = QLabel("Idle")
        layout.addWidget(self.bar)
        layout.addWidget(self.status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        layout.addWidget(self.log_view, 1)

    # subclasses override
    def build_opts(self) -> dict:
        raise NotImplementedError

    def job_fn(self):
        raise NotImplementedError

    def _start(self):
        try:
            opts = self.build_opts()
        except ValueError as exc:
            QMessageBox.warning(self, "Missing input", str(exc))
            return
        self.worker = Worker(self.job_fn(), opts)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.done.connect(self._on_done)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.bar.setValue(0)
        self.status.setText("Starting…")
        self.log_view.appendPlainText("--- job started ---")
        self.worker.start()

    def _cancel(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.status.setText("Cancelling…")

    def _on_progress(self, fraction: float, msg: str):
        self.bar.setValue(int(fraction * 1000))
        self.status.setText(msg)

    def _on_done(self, ok: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText(msg)
        self.log_view.appendPlainText(f"--- {msg} ---")
        if ok:
            self.bar.setValue(1000)
