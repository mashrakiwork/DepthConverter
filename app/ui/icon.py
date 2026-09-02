"""The app icon, with every size Windows needs for the taskbar and Alt-Tab."""

from pathlib import Path

from PySide6.QtGui import QIcon

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def app_icon() -> QIcon:
    icon = QIcon()
    # The .ico carries 16-256px variants; the .png covers any larger request
    # (Alt-Tab / task view on high-DPI screens).
    ico = ASSETS / "icon.ico"
    if ico.exists():
        icon.addFile(str(ico))
    png = ASSETS / "icon.png"
    if png.exists():
        icon.addFile(str(png))
    return icon
