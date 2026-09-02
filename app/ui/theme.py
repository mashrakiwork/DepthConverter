"""Global look-and-feel: readable control sizes that scale with a user zoom.

Qt's default Fusion metrics are tiny on high-DPI monitors, so every control
gets an explicit minimum height and font size derived from one scale factor
the user can change from the status bar.
"""

from PySide6.QtWidgets import QApplication

SCALES = [80, 90, 100, 110, 125, 150, 175, 200]
DEFAULT_SCALE = 100


def stylesheet(scale_percent: int = DEFAULT_SCALE) -> str:
    """Qt stylesheet sized for `scale_percent` (100 = the comfortable default)."""
    k = scale_percent / 100.0
    pt = round(10 * k, 1)          # base font size
    row = round(30 * k)            # height of a single-line control
    small = round(26 * k)
    pad = round(6 * k)
    return f"""
    QWidget {{ font-size: {pt}pt; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton {{
        min-height: {row}px;
        padding: {round(2 * k)}px {pad}px;
    }}
    QLineEdit, QComboBox {{ min-width: {round(160 * k)}px; }}
    QSpinBox, QDoubleSpinBox {{ min-width: {round(120 * k)}px; }}
    QComboBox QAbstractItemView {{ min-height: {small}px; }}
    QCheckBox {{ spacing: {pad}px; min-height: {small}px; }}
    QCheckBox::indicator {{
        width: {round(16 * k)}px; height: {round(16 * k)}px;
    }}
    QGroupBox {{
        font-weight: 600;
        border: 1px solid palette(mid);
        border-radius: {round(6 * k)}px;
        margin-top: {round(14 * k)}px;
        padding: {round(12 * k)}px {pad}px {pad}px {pad}px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: {round(10 * k)}px;
        padding: 0 {round(4 * k)}px;
    }}
    QTabBar::tab {{
        min-height: {small}px;
        padding: {pad}px {round(14 * k)}px;
    }}
    QProgressBar {{ min-height: {small}px; text-align: center; }}
    QPlainTextEdit {{ font-size: {round(9 * k, 1)}pt; }}
    QToolTip {{ font-size: {round(9.5 * k, 1)}pt; }}
    """


def apply_scale(scale_percent: int) -> None:
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(stylesheet(scale_percent))
