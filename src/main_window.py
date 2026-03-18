from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as Icon
from qfluentwidgets import FluentWindow, NavigationItemPosition, SplashScreen

from src.gpu_utils.cuda_check import get_device_info
from src.interfaces.bad_pixel_interface import BadPixelInterface
from src.interfaces.blackline_interface import BlacklineInterface
from src.interfaces.file_rename_interface import FileRenameInterface
from src.interfaces.image_process_interface import ImageProcessInterface
from src.interfaces.lifton2019_interface import Lifton2019Interface
from src.interfaces.oof_ct_sim_interface import OOFCTSimulationInterface
from src.interfaces.ring_artifact_interface import RingArtifactInterface
from src.interfaces.ui_theme import apply_app_theme, apply_interface_theme
from src.interfaces.wave_speed_interface import WaveSpeedInterface


def create_icon(icon_path: str) -> QIcon:
    return QIcon(icon_path)


class HomeInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homeInterface")
        apply_interface_theme(self)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(26, 20, 26, 20)
        layout.setSpacing(16)

        title_label = QLabel("GPU 加速 CT 图像处理软件", self)
        title_label.setStyleSheet(
            "font-size: 40px; font-weight: 700; color: #1f6fba; padding: 10px 0;"
        )
        title_label.setAlignment(Qt.AlignCenter)

        content = """
        <div style="
            background-color: #eef3fb;
            border-radius: 12px;
            padding: 24px;
            margin: 8px;
        ">
            <h3 style="
                color: #12304f;
                font-size: 28px;
                margin-top: 0;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 2px solid #1f6fba;
            ">项目功能总览</h3>

            <div style="
                background: white;
                padding: 14px 18px;
                border-radius: 8px;
                border-left: 4px solid #1f6fba;
                margin-bottom: 10px;
                color: #2f3f52;
                font-size: 18px;
                line-height: 1.6;
            ">
                <b>主程序 (main.py)</b><br>
                图像处理、黑线去除、环形伪影后处理、多点平场校正、坏点掩膜与修复、波阵面传播速度分析、批量重命名、超视野 CT 模拟。
            </div>

            <div style="
                background: white;
                padding: 14px 18px;
                border-radius: 8px;
                border-left: 4px solid #1d9c66;
                margin-bottom: 10px;
                color: #2f3f52;
                font-size: 18px;
                line-height: 1.6;
            ">
                <b>重构程序 (main_recon.py)</b><br>
                独立 CBCT 重构入口，支持 FDK / SIRT3D / CGLS3D、COR 自动搜索与子进程执行。
            </div>

            <div style="
                background: white;
                padding: 14px 18px;
                border-radius: 8px;
                border-left: 4px solid #f08b2b;
                margin-bottom: 10px;
                color: #2f3f52;
                font-size: 18px;
                line-height: 1.6;
            ">
                <b>GPU 说明</b><br>
                检测到 CUDA 时自动启用 GPU 加速；未检测到可用 GPU 或依赖时自动回退 CPU，保证功能可用。
            </div>

            <div style="
                text-align: center;
                margin-top: 6px;
                padding: 12px;
                background: white;
                border-radius: 8px;
                border: 1px dashed #b6c5da;
                color: #3d4d62;
                font-size: 16px;
            ">
                从左侧导航栏选择功能模块即可开始。
            </div>
        </div>
        """

        content_label = QLabel(content, self)
        content_label.setWordWrap(True)
        content_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(content_label)

        device_info = get_device_info()
        if device_info["available"]:
            status_text = f"""
            <div style="
                background-color: #dff5e8;
                border: 1px solid #b8e3c8;
                border-radius: 8px;
                padding: 10px 14px;
                margin: 4px;
                color: #15553b;
            ">
                <b>GPU 加速已启用</b><br>
                显卡: {device_info["name"]}<br>
                计算能力: {device_info["compute_capability"]}<br>
                显存: {device_info["total_memory"]} GB
            </div>
            """
        else:
            status_text = """
            <div style="
                background-color: #fff4d6;
                border: 1px solid #ffd281;
                border-radius: 8px;
                padding: 10px 14px;
                margin: 4px;
                color: #7f5a1a;
            ">
                <b>当前为 CPU 模式</b><br>
                未检测到可用的 NVIDIA GPU 或 CUDA 运行环境，将使用 CPU 执行处理流程。
            </div>
            """

        status_label = QLabel(status_text, self)
        status_label.setWordWrap(True)
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        layout.addStretch(1)


class MainWindow(FluentWindow):
    windowHidden = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.__init_window()

        self.homeInterface = HomeInterface(self)
        self.imageProcessInterface = ImageProcessInterface(self)
        self.blacklineInterface = BlacklineInterface(self)
        self.ringArtifactInterface = RingArtifactInterface(self)
        self.lifton2019Interface = Lifton2019Interface(self)
        self.badPixelInterface = BadPixelInterface(self)
        self.fileRenameInterface = FileRenameInterface(self)
        self.waveSpeedInterface = WaveSpeedInterface(self)
        self.oofCTSimulationInterface = OOFCTSimulationInterface(self)

        self.__init_navigation()

        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.__finish_initialization()

    def __init_window(self):
        self.setMinimumSize(1500, 960)
        self.setWindowIcon(QIcon("./resource/icons/icon2.png"))
        self.setWindowTitle("GPU 加速 CT 图像处理软件")

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        self.show()
        QApplication.processEvents()

    def __init_navigation(self):
        pos = NavigationItemPosition.TOP
        self.addSubInterface(self.homeInterface, Icon.HOME, "首页", pos)

        pos = NavigationItemPosition.SCROLL
        self.addSubInterface(
            self.imageProcessInterface,
            create_icon("./resource/icons/dataset.png"),
            "图像处理",
            pos,
        )
        self.addSubInterface(
            self.blacklineInterface,
            create_icon("./resource/icons/Googlefile.png"),
            "黑线去除",
            pos,
        )
        self.addSubInterface(
            self.ringArtifactInterface,
            create_icon("./resource/icons/addnoise.png"),
            "环形伪影后处理",
            pos,
        )
        self.addSubInterface(
            self.lifton2019Interface,
            create_icon("./resource/icons/resampletime.png"),
            "Lifton2019 多点平场",
            pos,
        )
        self.addSubInterface(
            self.badPixelInterface,
            create_icon("./resource/icons/change.png"),
            "坏点掩膜",
            pos,
        )
        self.addSubInterface(
            self.fileRenameInterface,
            create_icon("./resource/icons/rename.png"),
            "文件重命名",
            pos,
        )
        self.addSubInterface(
            self.waveSpeedInterface,
            create_icon("./resource/icons/progress.png"),
            "波阵面速度",
            pos,
        )
        self.addSubInterface(
            self.oofCTSimulationInterface,
            create_icon("./resource/icons/addnoise.png"),
            "超视野 CT 模拟",
            pos,
        )

        pos = NavigationItemPosition.BOTTOM
        self.addSubInterface(self.homeInterface, Icon.SETTING, "设置", pos)
        self.navigationInterface.setExpandWidth(270)

    def __finish_initialization(self):
        self.splashScreen.finish()
        self.switchTo(self.homeInterface)

    def switchTo(self, widget):
        self.stackedWidget.setCurrentWidget(widget, popOut=False)

    def closeEvent(self, event):
        self.windowHidden.emit(True)
        super().closeEvent(event)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    apply_app_theme(app)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
