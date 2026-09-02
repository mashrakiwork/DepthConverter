import os
import sys


def main() -> int:
    # Fully local: no telemetry leaves this machine.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    # Keep the Depth Anything 3 package quiet (per-batch timing spam, gsplat warning).
    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
    # Triton doesn't exist on Windows; stop xformers from probing for it.
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

    if sys.platform == "win32":
        # Give the process its own taskbar identity so Windows shows our icon
        # (instead of grouping under the generic pythonw.exe one).
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "DepthConverter.DepthConverter")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Crisp scaling on high-DPI / mixed-DPI monitor setups.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("DepthConverter")
    app.setOrganizationName("DepthConverter")
    app.setStyle("Fusion")

    from app.config import Config
    from app.ui.icon import app_icon
    from app.ui.theme import DEFAULT_SCALE, apply_scale

    app.setWindowIcon(app_icon())
    apply_scale(int(Config().get("ui.scale", DEFAULT_SCALE)))

    from app.ui.main_window import MainWindow

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
