from __future__ import annotations

from typing import Tuple

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QComboBox, QLayout, QPushButton, QWidget


APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f3f6fb;
    color: #1e293b;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d8e0ec;
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #0f4c81;
}

QLabel {
    color: #243447;
}

QLineEdit,
QTextEdit,
QPlainTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #c9d4e5;
    border-radius: 8px;
    min-height: 34px;
    padding: 4px 8px;
    selection-background-color: #2a7de1;
}

QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #2a7de1;
}

QTextEdit {
    background-color: #0f172a;
    color: #dbe7ff;
    border: 1px solid #1f2a44;
    border-radius: 8px;
}

QPushButton {
    background-color: #eef3fb;
    color: #12304f;
    border: 1px solid #c8d6ea;
    border-radius: 8px;
    min-height: 34px;
    padding: 6px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #e1ecfb;
}

QPushButton:pressed {
    background-color: #d5e5fb;
}

QPushButton:disabled {
    background-color: #f2f4f7;
    color: #9aa7ba;
    border-color: #d7deea;
}

QPushButton[role="primary"] {
    background-color: #1f6fba;
    color: #ffffff;
    border: 1px solid #165e9d;
}

QPushButton[role="primary"]:hover {
    background-color: #2a7de1;
}

QPushButton[role="success"] {
    background-color: #1d9c66;
    color: #ffffff;
    border: 1px solid #178553;
}

QPushButton[role="danger"] {
    background-color: #d64545;
    color: #ffffff;
    border: 1px solid #b43838;
}

QHeaderView::section {
    background-color: #e9f0fb;
    color: #1d3553;
    border: none;
    border-right: 1px solid #d4deef;
    border-bottom: 1px solid #d4deef;
    padding: 8px 6px;
    font-weight: 600;
}

QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f8fbff;
    border: 1px solid #d3deee;
    border-radius: 8px;
    gridline-color: #e2e9f4;
}

QProgressBar {
    border: 1px solid #c7d5e9;
    border-radius: 7px;
    background: #eef3fb;
    text-align: center;
    min-height: 20px;
}

QProgressBar::chunk {
    background-color: #1f6fba;
    border-radius: 6px;
}

QDockWidget {
    border: 1px solid #d3deee;
}

QDockWidget::title {
    background-color: #e9f0fb;
    color: #1d3553;
    padding: 6px 8px;
    text-align: center;
    font-weight: 600;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #c2d2e9;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #9eb6d8;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background: #c2d2e9;
    border-radius: 5px;
    min-width: 20px;
}

QTreeView {
    background-color: #ffffff;
    border: 1px solid #d3deee;
    border-radius: 8px;
}

QTreeView::item:selected {
    background-color: #d8e9ff;
    color: #133960;
}
"""


def _pick_font_family() -> str:
    available = set(QFontDatabase().families())
    preferred = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Segoe UI",
    ]
    for name in preferred:
        if name in available:
            return name
    return "Arial"


class _ComboBoxWheelGuard(QObject):
    """Prevent accidental combo-box changes on mouse wheel hover."""

    def eventFilter(self, obj, event):
        if isinstance(obj, QComboBox):
            if event.type() == QEvent.MouseButtonPress:
                obj.setProperty("_wheel_combo_active", True)
                return False

            if event.type() == QEvent.FocusOut:
                obj.setProperty("_wheel_combo_active", False)
                return False

            if event.type() == QEvent.Wheel:
                popup_visible = obj.view().isVisible() if obj.view() is not None else False
                wheel_enabled = bool(obj.property("_wheel_combo_active"))
                if popup_visible or (wheel_enabled and obj.hasFocus()):
                    return False
                event.ignore()
                return True

        return super().eventFilter(obj, event)


def _install_combobox_wheel_guard(app: QApplication) -> None:
    guard = getattr(app, "_combo_wheel_guard", None)
    if guard is None:
        guard = _ComboBoxWheelGuard(app)
        app._combo_wheel_guard = guard
        app.installEventFilter(guard)


def apply_app_theme(app: QApplication) -> None:
    app.setFont(QFont(_pick_font_family(), 10))
    app.setStyleSheet(APP_STYLESHEET)
    _install_combobox_wheel_guard(app)


def apply_interface_theme(widget: QWidget) -> None:
    widget.setAttribute(Qt.WA_StyledBackground, True)


def set_button_role(button: QPushButton, role: str) -> None:
    button.setProperty("role", role)
    style = button.style()
    style.unpolish(button)
    style.polish(button)
    button.update()


def tune_layout(layout: QLayout, margins: Tuple[int, int, int, int] = (12, 12, 12, 12), spacing: int = 10) -> None:
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
