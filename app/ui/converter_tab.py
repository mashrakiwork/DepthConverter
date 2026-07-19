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
        self.orig_pick.setToolTip("The original 2D content: one video file, or a "
                                  "folder of images.")
        self.depth_pick = PathPicker(cfg, "sbs.depth", "any",
                                     "Matching depth video file, or folder of depth images")
        self.depth_pick.setToolTip("The matching depth content made in the Depth tab: "
                                   "the *_depth video, or the folder with *_depth.png "
                                   "images.")
        self.output_pick = PathPicker(cfg, "sbs.output_dir", "dir",
                                      "Where the SBS output is written")
        self.output_pick.setToolTip("Folder where the finished 3D (SBS) files are "
                                    "written.")
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
        self.divergence.setToolTip(
            "How strong the 3D effect is - how far apart the left/right eye "
            "views are pushed (as % of the frame width). More = things pop out "
            "of the screen harder, but too much strains the eyes and causes "
            "double vision. Recommended: 1.0-1.5. Start at 1.2 and only raise "
            "it if the 3D feels flat in your headset.")
        self.auto_conv_check = QCheckBox("Auto convergence - keep the subject on the "
                                         "screen plane (recommended)")
        self.auto_conv_check.setToolTip(
            "Automatically tracks the main subject's depth every frame and "
            "places it exactly on the virtual screen, so your eyes fuse the "
            "image easily. This is the single biggest comfort factor - keep "
            "it ON unless you want a fixed artistic depth placement.")
        self.auto_conv_check.setChecked(bool(cfg.get("sbs.auto_convergence", True)))
        self.convergence = QDoubleSpinBox()
        self.convergence.setRange(0.0, 1.0)
        self.convergence.setSingleStep(0.05)
        self.convergence.setValue(float(cfg.get("sbs.convergence", 0.5)))
        self.convergence.setToolTip(
            "Only used when auto convergence is off. Which depth (0 = farthest, "
            "1 = nearest) sits exactly on the screen: nearer content pops out, "
            "farther content goes behind. Recommended: 0.5.")
        self.convergence.setEnabled(not self.auto_conv_check.isChecked())
        self.auto_conv_check.toggled.connect(
            lambda on: self.convergence.setEnabled(not on))
        self.aa_check = QCheckBox("Anti-aliasing (smooth object outlines)")
        self.aa_check.setToolTip(
            "Renders the stereo warp at a higher internal resolution and "
            "filters it back down, removing jagged pixel stairs on object "
            "perimeters. Recommended: ON.")
        self.aa_check.setChecked(bool(cfg.get("sbs.antialias", True)))
        self.aa_quality = QComboBox()
        self.aa_quality.setToolTip(
            "Anti-aliasing quality: how much higher the internal rendering "
            "resolution is. Higher = smoother outlines but more VRAM and "
            "slower. Recommended: 2x; use 3x-4x if you still notice jagged "
            "edges (needs a strong GPU at 4K).")
        self.aa_quality.addItem("2x (balanced)", 2)
        self.aa_quality.addItem("3x (high)", 3)
        self.aa_quality.addItem("4x (ultra)", 4)
        idx = self.aa_quality.findData(int(cfg.get("sbs.aa_quality", 2)))
        self.aa_quality.setCurrentIndex(max(idx, 0))
        self.aa_quality.setEnabled(self.aa_check.isChecked())
        self.aa_check.toggled.connect(self.aa_quality.setEnabled)

        self.smooth_check = QCheckBox("Smooth depth edges (reduces tearing/shimmer)")
        self.smooth_check.setToolTip(
            "Lightly smooths the depth map before warping, removing blocky "
            "stair-steps that video compression leaves at object edges. "
            "Recommended: ON.")
        self.smooth_check.setChecked(bool(cfg.get("sbs.smooth_depth", True)))
        self.layout_combo = QComboBox()
        self.layout_combo.setToolTip(
            "Full SBS: output is twice the source width, each eye at full "
            "resolution - best quality, recommended. Half SBS: same width as "
            "the source, each eye squeezed to half - smaller file, for players "
            "or displays that only accept half SBS.")
        self.layout_combo.addItems(["Full SBS (2x width)", "Half SBS (same width)"])
        self.layout_combo.setCurrentIndex(int(cfg.get("sbs.half", 0)))
        self.invert_check = QCheckBox("Depth is inverted (white = far)")
        self.invert_check.setToolTip(
            "Only tick this if your depth content has the opposite convention "
            "(white = far). With depth made by this app, leave it OFF.")
        self.invert_check.setChecked(bool(cfg.get("sbs.invert_depth", False)))
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Where the stereo warping runs. Auto picks your "
                                     "NVIDIA GPU when available - recommended.")
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.device_combo.setCurrentText(cfg.get("sbs.device", "Auto"))

        sgrid.addWidget(QLabel("3D strength (pop-out):"), 0, 0)
        sgrid.addWidget(self.divergence, 0, 1)
        sgrid.addWidget(self.auto_conv_check, 1, 0, 1, 2)
        sgrid.addWidget(QLabel("Manual convergence:"), 2, 0)
        sgrid.addWidget(self.convergence, 2, 1)
        sgrid.addWidget(self.aa_check, 3, 0)
        sgrid.addWidget(self.aa_quality, 3, 1)
        sgrid.addWidget(self.smooth_check, 4, 0, 1, 2)
        sgrid.addWidget(QLabel("Output layout:"), 5, 0)
        sgrid.addWidget(self.layout_combo, 5, 1)
        sgrid.addWidget(QLabel("Device:"), 6, 0)
        sgrid.addWidget(self.device_combo, 6, 1)
        sgrid.addWidget(self.invert_check, 7, 0, 1, 2)
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
        self.cfg.set("sbs.antialias", self.aa_check.isChecked())
        self.cfg.set("sbs.aa_quality", self.aa_quality.currentData())
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
            "aa_supersample": (self.aa_quality.currentData()
                               if self.aa_check.isChecked() else 1),
            "smooth_depth": self.smooth_check.isChecked(),
            "half_sbs": half,
            "invert_depth": self.invert_check.isChecked(),
            "device": self.device_combo.currentText().lower(),
            **self.encoding.opts(),
        }

    def job_fn(self):
        from app.core.pipeline import run_sbs_job

        return run_sbs_job
