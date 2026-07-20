"""Window 0: AI-upscale images or a video before depth/SBS conversion."""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout,
)

from app.core.upscale import UPSCALE_MODELS, parse_model_spec
from app.ui.common import MEDIA_FILTER, EncodingGroup, JobTab, PathPicker


class UpscaleTab(JobTab):
    def __init__(self, cfg):
        super().__init__(cfg)
        layout = QVBoxLayout(self)

        folders = QGroupBox("Input / output")
        fgrid = QGridLayout(folders)
        self.input_pick = PathPicker(cfg, "upscale.input", "any",
                                     "A single video/image file, or a folder of images",
                                     file_filter=MEDIA_FILTER)
        self.input_pick.setToolTip("What to upscale: pick one video file, one image "
                                   "file, or a whole folder of images.")
        self.output_pick = PathPicker(cfg, "upscale.output_dir", "dir",
                                      "Where the upscaled output is written")
        self.output_pick.setToolTip("Folder where the upscaled files are written.")
        fgrid.addWidget(QLabel("Input (file or folder):"), 0, 0)
        fgrid.addWidget(self.input_pick, 0, 1)
        fgrid.addWidget(QLabel("Output folder:"), 1, 0)
        fgrid.addWidget(self.output_pick, 1, 1)
        fgrid.setColumnStretch(1, 1)
        layout.addWidget(folders)

        model_box = QGroupBox("Upscale model")
        mgrid = QGridLayout(model_box)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for label, repo, filename in UPSCALE_MODELS:
            self.model_combo.addItem(label, (repo, filename))
        saved = cfg.get("upscale.model")
        idx = self.model_combo.findData(tuple(saved)) if saved else 0
        if idx >= 0:
            self.model_combo.setCurrentIndex(max(idx, 0))
        elif saved:
            self.model_combo.setEditText(" :: ".join(saved))
        self.model_combo.setToolTip(
            "Which AI model does the upscaling. Real-ESRGAN x4plus is the "
            "best all-rounder for photos/video; UltraSharp resolves finer "
            "detail; AnimeSharp is tuned for drawn content. You can also type "
            "any HuggingFace-hosted weights as 'repo_id :: filename.pth'. "
            "Models download once and are cached locally.")

        self.scale_combo = QComboBox()
        self.scale_combo.setToolTip(
            "How much larger the output is (both width and height). The "
            "model's native scale is chained or downsampled to hit this "
            "exactly, so the aspect ratio never changes.")
        for s in (2, 3, 4):
            self.scale_combo.addItem(f"{s}x", s)
        idx = self.scale_combo.findData(int(cfg.get("upscale.scale", 2)))
        self.scale_combo.setCurrentIndex(max(idx, 0))

        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Where the model runs. Auto picks your NVIDIA "
                                     "GPU when available - recommended (CPU is very "
                                     "slow).")
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.device_combo.setCurrentText(cfg.get("upscale.device", "Auto"))
        self.fp16_check = QCheckBox("FP16 (half precision - faster, less VRAM)")
        self.fp16_check.setToolTip("Runs the model at 16-bit precision: about twice "
                                   "as fast and half the VRAM, with no visible "
                                   "quality loss. Recommended: ON.")
        self.fp16_check.setChecked(bool(cfg.get("upscale.fp16", True)))
        self.deborder_check = QCheckBox("Remove black borders (crop letterbox bars)")
        self.deborder_check.setToolTip(
            "Detects constant black bars around the content (letterbox / "
            "pillarbox) and crops them away before upscaling, so no GPU time "
            "is wasted on empty bars. Does nothing if there are none. "
            "Recommended: ON.")
        self.deborder_check.setChecked(bool(cfg.get("upscale.remove_borders", True)))

        self.delete_model_btn = QPushButton("Delete downloaded model…")
        self.delete_model_btn.setToolTip(
            "Removes the selected model's files from your disk to free space. "
            "The model stays in the list and will simply re-download if you "
            "use it again.")
        self.delete_model_btn.clicked.connect(self._delete_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.delete_model_btn)

        mgrid.addWidget(QLabel("Model:"), 0, 0)
        mgrid.addLayout(model_row, 0, 1)
        mgrid.addWidget(QLabel("Scale:"), 1, 0)
        mgrid.addWidget(self.scale_combo, 1, 1)
        mgrid.addWidget(QLabel("Device:"), 2, 0)
        mgrid.addWidget(self.device_combo, 2, 1)
        mgrid.addWidget(self.fp16_check, 3, 0, 1, 2)
        mgrid.addWidget(self.deborder_check, 4, 0, 1, 2)
        mgrid.setColumnStretch(1, 1)
        layout.addWidget(model_box)

        note = QLabel("Tip: upscale BEFORE the Depth tab for the best 3D quality - "
                      "depth models and the SBS warp both benefit from the extra "
                      "resolution. Frames are processed in overlapping tiles, so "
                      "even 4K input fits in VRAM.")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.encoding = EncodingGroup(cfg, "upscale.")
        layout.addWidget(self.encoding)

        self._add_run_controls(layout)

    def _model_spec(self) -> tuple[str, str]:
        idx = self.model_combo.currentIndex()
        text = self.model_combo.currentText().strip()
        if idx >= 0 and self.model_combo.itemText(idx) == text:
            return self.model_combo.itemData(idx)
        return parse_model_spec(text)

    def _delete_model(self):
        from app.core.models import cached_size_bytes, delete_cached

        try:
            repo_id, _ = self._model_spec()
        except ValueError:
            return
        size = cached_size_bytes(repo_id)
        if size is None:
            QMessageBox.information(self, "Not downloaded",
                                    f"'{repo_id}' is not on your disk - nothing "
                                    f"to delete.")
            return
        gb = size / (1024 ** 3)
        answer = QMessageBox.question(
            self, "Delete model from disk?",
            f"Delete '{repo_id}' ({gb:.2f} GB) from your disk?\n\n"
            f"It stays in the list and will re-download if you use it again.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            freed = delete_cached(repo_id)
            self.log_view.appendPlainText(
                f"Deleted '{repo_id}' - freed {freed / (1024 ** 3):.2f} GB.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Delete failed", str(exc))

    def stage_opts(self) -> dict:
        """Everything except the input/output paths (also used by Run all)."""
        repo_id, filename = self._model_spec()  # raises ValueError if malformed
        self.cfg.set("upscale.model", [repo_id, filename])
        self.cfg.set("upscale.scale", self.scale_combo.currentData())
        self.cfg.set("upscale.device", self.device_combo.currentText())
        self.cfg.set("upscale.fp16", self.fp16_check.isChecked())
        self.cfg.set("upscale.remove_borders", self.deborder_check.isChecked())
        return {
            "repo_id": repo_id,
            "filename": filename,
            "scale": self.scale_combo.currentData(),
            "device": self.device_combo.currentText().lower(),
            "fp16": self.fp16_check.isChecked(),
            "remove_borders": self.deborder_check.isChecked(),
            **self.encoding.opts(),
        }

    def build_opts(self) -> dict:
        if not self.input_pick.path():
            raise ValueError("Choose an input file or folder.")
        if not self.output_pick.path():
            raise ValueError("Choose an output folder.")
        return {
            "input_path": self.input_pick.path(),
            "output_dir": self.output_pick.path(),
            **self.stage_opts(),
        }

    def job_fn(self):
        from app.core.pipeline import run_upscale_job

        return run_upscale_job
