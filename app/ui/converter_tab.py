"""Window 2: combine original + depth content into side-by-side 3D (SBS)."""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel,
    QVBoxLayout,
)

from app.ui.common import EncodingGroup, JobTab, PathPicker


class ConverterTab(JobTab):
    def __init__(self, cfg):
        super().__init__(cfg)
        layout = QVBoxLayout(self)

        paths = QGroupBox("Content")
        pgrid = QGridLayout(paths)
        self.orig_pick = PathPicker(cfg, "sbs.original", "any",
                                    "Original video file, or folder of images")
        self.depth_pick = PathPicker(cfg, "sbs.depth", "any",
                                     "Matching depth video file, or folder of depth images")
        self.output_pick = PathPicker(cfg, "sbs.output_dir", "dir",
                                      "Where the SBS output is written")
        pgrid.addWidget(QLabel("Original:"), 0, 0)
        pgrid.addWidget(self.orig_pick, 0, 1)
        pgrid.addWidget(QLabel("Depth:"), 1, 0)
        pgrid.addWidget(self.depth_pick, 1, 1)
        pgrid.addWidget(QLabel("Output folder:"), 2, 0)
        pgrid.addWidget(self.output_pick, 2, 1)
        pgrid.setColumnStretch(1, 1)
        layout.addWidget(paths)

        # Old builds used per-eye divergence with default 2.0 (too strong, caused
        # double vision); migrate that stale default to the new comfortable one.
        saved_div = float(cfg.get("sbs.divergence", 1.2))
        if saved_div == 2.0:
            saved_div = 1.2

        stereo = QGroupBox("Stereo settings")
        sgrid = QGridLayout(stereo)
        self.divergence = QDoubleSpinBox()
        self.divergence.setRange(0.1, 4.0)
        self.divergence.setSingleStep(0.1)
        self.divergence.setValue(saved_div)
        self.divergence.setSuffix(" % of width (total)")
        self.divergence.setToolTip("Total disparity budget between the two eyes. "
                                   "1-1.5% is comfortable in VR; more pops harder "
                                   "but risks eye strain and double vision.")
        self.auto_conv_check = QCheckBox("Auto convergence - keep the subject on the "
                                         "screen plane (recommended)")
        self.auto_conv_check.setChecked(bool(cfg.get("sbs.auto_convergence", True)))
        self.convergence = QDoubleSpinBox()
        self.convergence.setRange(0.0, 1.0)
        self.convergence.setSingleStep(0.05)
        self.convergence.setValue(float(cfg.get("sbs.convergence", 0.5)))
        self.convergence.setToolTip("Manual: depth level that sits exactly on the "
                                    "screen plane. Nearer pops out, farther goes in.")
        self.convergence.setEnabled(not self.auto_conv_check.isChecked())
        self.auto_conv_check.toggled.connect(
            lambda on: self.convergence.setEnabled(not on))
        self.smooth_check = QCheckBox("Smooth depth edges (reduces tearing/shimmer)")
        self.smooth_check.setChecked(bool(cfg.get("sbs.smooth_depth", True)))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Full SBS (2x width)", "Half SBS (same width)"])
        self.layout_combo.setCurrentIndex(int(cfg.get("sbs.half", 0)))
        self.invert_check = QCheckBox("Depth is inverted (white = far)")
        self.invert_check.setChecked(bool(cfg.get("sbs.invert_depth", False)))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.device_combo.setCurrentText(cfg.get("sbs.device", "Auto"))

        sgrid.addWidget(QLabel("3D strength (disparity):"), 0, 0)
        sgrid.addWidget(self.divergence, 0, 1)
        sgrid.addWidget(self.auto_conv_check, 1, 0, 1, 2)
        sgrid.addWidget(QLabel("Manual convergence:"), 2, 0)
        sgrid.addWidget(self.convergence, 2, 1)
        sgrid.addWidget(self.smooth_check, 3, 0, 1, 2)
        sgrid.addWidget(QLabel("Output layout:"), 4, 0)
        sgrid.addWidget(self.layout_combo, 4, 1)
        sgrid.addWidget(QLabel("Device:"), 5, 0)
        sgrid.addWidget(self.device_combo, 5, 1)
        sgrid.addWidget(self.invert_check, 6, 0, 1, 2)
        sgrid.setColumnStretch(1, 1)
        layout.addWidget(stereo)

        note = QLabel("VR playback: outputs are tagged _Full_SBS_LRF / _Half_SBS_LR so "
                      "players auto-detect the layout. In Skybox, if it still looks "
                      "squeezed or doubled, manually set the video format to "
                      "\"Full SBS\" (for 2x-width files).")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.encoding = EncodingGroup(cfg, "sbs.")
        layout.addWidget(self.encoding)

        self._add_run_controls(layout)

    def build_opts(self) -> dict:
        if not self.orig_pick.path():
            raise ValueError("Choose the original content (video file or image folder).")
        if not self.depth_pick.path():
            raise ValueError("Choose the depth content (video file or image folder).")
        if not self.output_pick.path():
            raise ValueError("Choose an output folder.")
        half = self.layout_combo.currentIndex() == 1
        self.cfg.set("sbs.divergence", self.divergence.value())
        self.cfg.set("sbs.convergence", self.convergence.value())
        self.cfg.set("sbs.auto_convergence", self.auto_conv_check.isChecked())
        self.cfg.set("sbs.smooth_depth", self.smooth_check.isChecked())
        self.cfg.set("sbs.half", self.layout_combo.currentIndex())
        self.cfg.set("sbs.invert_depth", self.invert_check.isChecked())
        self.cfg.set("sbs.device", self.device_combo.currentText())
        return {
            "original": self.orig_pick.path(),
            "depth": self.depth_pick.path(),
            "output_dir": self.output_pick.path(),
            "divergence": self.divergence.value(),
            "convergence": self.convergence.value(),
            "auto_convergence": self.auto_conv_check.isChecked(),
            "smooth_depth": self.smooth_check.isChecked(),
            "half_sbs": half,
            "invert_depth": self.invert_check.isChecked(),
            "device": self.device_combo.currentText().lower(),
            **self.encoding.opts(),
        }

    def job_fn(self):
        from app.core.pipeline import run_sbs_job

        return run_sbs_job
