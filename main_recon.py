import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from src.interfaces.reconstruction_interface import ReconstructionInterface
from src.interfaces.ui_theme import apply_app_theme


def main():
    app = QApplication(sys.argv)
    apply_app_theme(app)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = ReconstructionInterface()
    window.setWindowIcon(QIcon("./resource/icons/icon2.png"))
    window.setWindowTitle("CBCT重构软件")
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
