"""Window 1: generate depth maps / depth video from a folder of images or one video."""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

from app.core.models import MODELS
from app.ui.common import MEDIA_FILTER, EncodingGroup, JobTab, PathPicker


class DepthTab(JobTab):
    def __init__(self, cfg):
        super().__init__(cfg)
        layout = QVBoxLayout(self)

        folders = QGroupBox("Input / output")
        fgrid = QGridLayout(folders)
        self.input_pick = PathPicker(cfg, "depth.input_dir", "any",
                                     "A single video/image file, or a folder of images "
                                     "(or one with a single video)",
                                     file_filter=MEDIA_FILTER)
        self.input_pick.setToolTip("What to compute depth for: pick one video file, "
                                   "one image file, or a whole folder of images.")
        self.output_pick = PathPicker(cfg, "depth.output_dir", "dir",
                                      "Where depth output is written")
        self.output_pick.setToolTip("Folder where the depth maps / depth video are "
                                    "written.")
        fgrid.addWidget(QLabel("Input (file or folder):"), 0, 0)
        fgrid.addWidget(self.input_pick, 0, 1)
        fgrid.addWidget(QLabel("Output folder:"), 1, 0)
        fgrid.addWidget(self.output_pick, 1, 1)
        fgrid.setColumnStretch(1, 1)
        layout.addWidget(folders)

        model_box = QGroupBox("Depth model")
        mgrid = QGridLayout(model_box)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for label, repo in MODELS:
            self.model_combo.addItem(label, repo)
        saved = cfg.get("depth.model")
        idx = self.model_combo.findData(saved) if saved else 0
        if idx >= 0:
            self.model_combo.setCurrentIndex(max(idx, 0))
        elif saved:
            self.model_combo.setEditText(saved)
        self.model_combo.setToolTip(
            "Which AI model estimates depth. Bigger = better quality, slower. "
            "Recommended: Depth Anything V3 Mono Large for best results, or "
            "V2 Small for quick previews. You can also type any HuggingFace "
            "depth-estimation repo id. Models download once and are cached "
            "locally.")

        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Where the model runs. Auto picks your NVIDIA "
                                     "GPU when available - recommended (CPU is very "
                                     "slow).")
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.device_combo.setCurrentText(cfg.get("depth.device", "Auto"))
        self.fp16_check = QCheckBox("FP16 (half precision - faster, less VRAM)")
        self.fp16_check.setToolTip("Runs the model at 16-bit precision: about twice "
                                   "as fast and half the VRAM, with no visible "
                                   "quality loss. Recommended: ON.")
        self.fp16_check.setChecked(bool(cfg.get("depth.fp16", True)))
        self.invert_check = QCheckBox("Invert depth (swap black/white if near/far look flipped)")
        self.invert_check.setToolTip("Flips the depth output. Normally OFF - only "
                                     "use if your depth output shows near things "
                                     "dark instead of bright.")
        self.invert_check.setChecked(bool(cfg.get("depth.invert", False)))
        convention = QLabel("Convention: WHITE = near, black = far. Only invert if "
                            "your depth output visibly shows the opposite.")
        convention.setWordWrap(True)

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
        mgrid.addWidget(QLabel("Device:"), 1, 0)
        mgrid.addWidget(self.device_combo, 1, 1)
        mgrid.addWidget(self.fp16_check, 2, 0, 1, 2)
        mgrid.addWidget(self.invert_check, 3, 0, 1, 2)
        mgrid.addWidget(convention, 4, 0, 1, 2)
        mgrid.setColumnStretch(1, 1)
        layout.addWidget(model_box)

        img_box = QGroupBox("Images as video")
        igrid = QGridLayout(img_box)
        self.slideshow_check = QCheckBox(
            "Also build a normal video + depth video from the images (slideshow)")
        self.slideshow_check.setToolTip(
            "For image-folder input: besides the depth PNGs, also builds a "
            "slideshow video of the originals plus a matching depth video, so "
            "you can convert them to a 3D video in the Converter tab.")
        self.slideshow_check.setChecked(bool(cfg.get("depth.images_to_video", False)))
        self.spf_spin = QDoubleSpinBox()
        self.spf_spin.setToolTip("How long each image stays on screen in the "
                                 "slideshow video.")
        self.spf_spin.setRange(0.1, 60.0)
        self.spf_spin.setSingleStep(0.5)
        self.spf_spin.setValue(float(cfg.get("depth.seconds_per_image", 2.0)))
        self.spf_spin.setSuffix(" s per image")
        igrid.addWidget(self.slideshow_check, 0, 0, 1, 2)
        igrid.addWidget(QLabel("Image duration:"), 1, 0)
        igrid.addWidget(self.spf_spin, 1, 1)
        igrid.setColumnStretch(1, 1)
        layout.addWidget(img_box)

        self.encoding = EncodingGroup(cfg, "depth.")
        layout.addWidget(self.encoding)

        self._add_run_controls(layout)

    def _delete_model(self):
        from app.core.models import cached_size_bytes, delete_cached

        model_id = self._model_id()
        if not model_id:
            return
        size = cached_size_bytes(model_id)
        if size is None:
            QMessageBox.information(self, "Not downloaded",
                                    f"'{model_id}' is not on your disk - nothing "
                                    f"to delete.")
            return
        gb = size / (1024 ** 3)
        answer = QMessageBox.question(
            self, "Delete model from disk?",
            f"Delete '{model_id}' ({gb:.2f} GB) from your disk?\n\n"
            f"It stays in the list and will re-download if you use it again.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            freed = delete_cached(model_id)
            self.log_view.appendPlainText(
                f"Deleted '{model_id}' - freed {freed / (1024 ** 3):.2f} GB.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Delete failed", str(exc))

    def _model_id(self) -> str:
        idx = self.model_combo.currentIndex()
        text = self.model_combo.currentText().strip()
        if idx >= 0 and self.model_combo.itemText(idx) == text:
            return self.model_combo.itemData(idx)
        return text

    def build_opts(self) -> dict:
        if not self.input_pick.path():
            raise ValueError("Choose an input file or folder.")
        if not self.output_pick.path():
            raise ValueError("Choose an output folder.")
        model_id = self._model_id()
        if not model_id:
            raise ValueError("Choose or type a depth model.")
        self.cfg.set("depth.model", model_id)
        self.cfg.set("depth.device", self.device_combo.currentText())
        self.cfg.set("depth.fp16", self.fp16_check.isChecked())
        self.cfg.set("depth.invert", self.invert_check.isChecked())
        self.cfg.set("depth.images_to_video", self.slideshow_check.isChecked())
        self.cfg.set("depth.seconds_per_image", self.spf_spin.value())
        return {
            "input_path": self.input_pick.path(),
            "output_dir": self.output_pick.path(),
            "model_id": model_id,
            "device": self.device_combo.currentText().lower(),
            "fp16": self.fp16_check.isChecked(),
            "invert": self.invert_check.isChecked(),
            "images_to_video": self.slideshow_check.isChecked(),
            "seconds_per_image": self.spf_spin.value(),
            **self.encoding.opts(),
        }

    def job_fn(self):
        from app.core.pipeline import run_depth_job

        return run_depth_job
