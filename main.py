import os
import sys

os.environ["TIFFFILE_USEIMAGEIO"] = "true"
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.interfaces.ui_theme import apply_app_theme
from src.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    apply_app_theme(app)

    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
