"""Run-all window: chain Upscale -> Depth -> Convert with one Start button.

Each stage is optional (checkbox) and takes its settings live from its own
tab - only the input/output here are used, and each stage's output feeds the
next stage automatically.
"""

from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QLabel, QVBoxLayout,
)

from app.ui.common import MEDIA_FILTER, JobTab, PathPicker


class PipelineTab(JobTab):
    def __init__(self, cfg, upscale_tab, depth_tab, converter_tab):
        super().__init__(cfg)
        self._tabs = {"upscale": upscale_tab, "depth": depth_tab,
                      "sbs": converter_tab}
        layout = QVBoxLayout(self)

        paths = QGroupBox("Input / output")
        pgrid = QGridLayout(paths)
        self.input_pick = PathPicker(cfg, "pipeline.input", "any",
                                     "The ORIGINAL content: a single video/image "
                                     "file, or a folder of images",
                                     file_filter=MEDIA_FILTER)
        self.input_pick.setToolTip("The original 2D content to run the selected "
                                   "stages on: one video file, one image file, or "
                                   "a whole folder of images.")
        self.output_pick = PathPicker(cfg, "pipeline.output_dir", "dir",
                                      "Where all outputs (including intermediate "
                                      "files) are written")
        self.output_pick.setToolTip("Every stage writes its output here; the final "
                                    "3D (SBS) files also end up in this folder.")
        pgrid.addWidget(QLabel("Input (file or folder):"), 0, 0)
        pgrid.addWidget(self.input_pick, 0, 1)
        pgrid.addWidget(QLabel("Output folder:"), 1, 0)
        pgrid.addWidget(self.output_pick, 1, 1)
        pgrid.setColumnStretch(1, 1)
        layout.addWidget(paths)

        stages = QGroupBox("Stages to run")
        sgrid = QVBoxLayout(stages)
        self.upscale_check = QCheckBox("1. Upscale (AI upscaling - untick if your "
                                       "content is already high resolution)")
        self.upscale_check.setChecked(bool(cfg.get("pipeline.do_upscale", True)))
        self.depth_check = QCheckBox("2. Depth (generate the depth maps)")
        self.depth_check.setChecked(bool(cfg.get("pipeline.do_depth", True)))
        self.sbs_check = QCheckBox("3. Convert (combine into 3D SBS for VR)")
        self.sbs_check.setChecked(bool(cfg.get("pipeline.do_sbs", True)))
        for c in (self.upscale_check, self.depth_check, self.sbs_check):
            sgrid.addWidget(c)
        note = QLabel("Each stage uses the settings from its own tab (model, "
                      "scale, 3D strength, encoder...) - set them there first. "
                      "Outputs flow automatically: Upscale feeds Depth, and "
                      "Convert pairs the (upscaled) original with the fresh "
                      "depth output.")
        note.setWordWrap(True)
        sgrid.addWidget(note)
        layout.addWidget(stages)

        self._add_run_controls(layout)

    def build_opts(self) -> dict:
        if not self.input_pick.path():
            raise ValueError("Choose an input file or folder.")
        if not self.output_pick.path():
            raise ValueError("Choose an output folder.")
        do = {key: check.isChecked() for key, check in
              (("upscale", self.upscale_check), ("depth", self.depth_check),
               ("sbs", self.sbs_check))}
        if not any(do.values()):
            raise ValueError("Tick at least one stage to run.")
        if do["sbs"] and not do["depth"]:
            raise ValueError(
                "Convert (SBS) needs the Depth stage ticked - it pairs the "
                "original with the depth output made in the same run. If you "
                "already have depth content, use the Converter tab directly.")
        for key, ticked in do.items():
            self.cfg.set(f"pipeline.do_{key}", ticked)
        opts = {
            "input_path": self.input_pick.path(),
            "output_dir": self.output_pick.path(),
            **{f"do_{key}": ticked for key, ticked in do.items()},
        }
        # Collect settings only from the tabs whose stage actually runs, so a
        # half-filled tab of an unticked stage can't block the pipeline.
        for key, ticked in do.items():
            if ticked:
                opts[key] = self._tabs[key].stage_opts()
        return opts

    def job_fn(self):
        from app.core.pipeline import run_pipeline_job

        return run_pipeline_job
