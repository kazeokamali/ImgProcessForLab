import sys
import os
os.environ['TIFFFILE_USEIMAGEIO'] = 'true'
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from src.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    font = QFont("Arial", 11)
    app.setFont(font)

    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
