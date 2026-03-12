from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QTextEdit, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as Icon
from qfluentwidgets import FluentWindow, NavigationItemPosition, SplashScreen
from src.interfaces.image_process_interface import ImageProcessInterface
from src.interfaces.blackline_interface import BlacklineInterface
from src.interfaces.ring_artifact_interface import RingArtifactInterface
from src.interfaces.lifton2019_interface import Lifton2019Interface
from src.interfaces.bad_pixel_interface import BadPixelInterface
from src.interfaces.file_rename_interface import FileRenameInterface
from src.interfaces.wave_speed_interface import WaveSpeedInterface
from src.gpu_utils.cuda_check import CUDA_AVAILABLE, get_device_info


def create_icon(icon_path):
    return QIcon(icon_path)


class HomeInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homeInterface")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("GPU加速CT图像处理软件", self)
        title_label.setStyleSheet("""
            font-size: 40px; 
            font-weight: bold; 
            color: #0078d4; 
            padding: 20px;
            margin-bottom: 10px;
        """)

        content = """
        <div style="
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 30px;
            margin: 20px;
        ">
            <h3 style="
                color: #333;
                font-size: 30px;
                margin-top: 0;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #0078d4;
            ">🎯 核心功能</h3>
            
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="
                    flex: 1;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #0078d4;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="
                        color: #0078d4;
                        font-size: 30px;
                        margin: 0 0 10px 0;
                    ">🖼️ 图像处理</h4>
                    <ul style="
                        color: #555;
                        font-size: 20px;
                        margin: 0;
                        padding-left: 20px;
                        list-style-position: outside;
                        text-align: left;
                    ">
                        <li>Raw转Tiff</li>
                        <li>图像裁剪</li>
                        <li>图像旋转</li>
                        <li>边缘锐化</li>
                        <li>取负对数</li>
                    </ul>
                </div>
                
                <div style="
                    flex: 1;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="
                        color: #28a745;
                        font-size: 30px;
                        margin: 0 0 10px 0;
                    ">🔍 黑线去除</h4>
                    <ul style="
                        color: #555;
                        font-size: 20px;
                        margin: 0;
                        padding-left: 20px;
                        list-style-position: outside;
                        text-align: left;
                    ">
                        <li>范围法去黑线</li>
                        <li>梯度法去黑线</li>
                        <li>背底处理</li>
                    </ul>
                </div>
            </div>
            
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: black;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
                text-align: center;
            ">
                <p style="
                    margin: 0;
                    font-size: 16px;
                    font-weight: 500;
                ">⚡ 所有图像处理均使用GPU加速（CuPy），处理速度提升10-100倍</p>
            </div>
            
            <div style="
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                background: white;
                border-radius: 8px;
                border: 2px dashed #ccc;
            ">
                <p style="
                    margin: 0;
                    color: #666;
                    font-size: 20px;
                ">👈 请从左侧导航栏选择功能模块开始使用</p>
            </div>
        </div>
        """
        content_label = QLabel(content, self)
        content_label.setWordWrap(True)
        content_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(content_label)
        
        device_info = get_device_info()
        if device_info['available']:
            status_text = f"""
            <div style="
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
                color: #155724;
            ">
                ✅ <b>GPU加速已启用</b><br>
                显卡: {device_info['name']}<br>
                计算能力: {device_info['compute_capability']}<br>
                显存: {device_info['total_memory']} GB
            </div>
            """
        else:
            status_text = f"""
            <div style="
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
                color: #856404;
            ">
                ⚠️ <b>CPU模式运行</b><br>
                未检测到NVIDIA GPU或CUDA驱动，将使用CPU处理图像。<br>
                如需GPU加速，请安装NVIDIA显卡驱动和CUDA 12.x。
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

        self.__initWindow()

        self.homeInterface = HomeInterface(self)
        self.imageProcessInterface = ImageProcessInterface(self)
        self.blacklineInterface = BlacklineInterface(self)
        self.ringArtifactInterface = RingArtifactInterface(self)
        self.lifton2019Interface = Lifton2019Interface(self)
        self.badPixelInterface = BadPixelInterface(self)
        self.fileRenameInterface = FileRenameInterface(self)
        self.waveSpeedInterface = WaveSpeedInterface(self)

        self.__initNavigation()

        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))

        self.__finishInitialization()

    def __initWindow(self):
        self.setMinimumSize(1700, 1200)
        self.setWindowIcon(QIcon("./resource/icons/icon2.png"))
        self.setWindowTitle("GPU加速CT图像处理软件")

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        self.show()
        QApplication.processEvents()

    def __initNavigation(self):
        pos = NavigationItemPosition.TOP

        self.addSubInterface(self.homeInterface, Icon.HOME, "主页", pos)

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

        pos = NavigationItemPosition.BOTTOM

        self.addSubInterface(
            self.homeInterface,
            Icon.SETTING,
            "Settings",
            pos,
        )

        self.navigationInterface.setExpandWidth(270)

    def __finishInitialization(self):
        self.splashScreen.finish()
        self.switchTo(self.homeInterface)

    def switchTo(self, widget):
        self.stackedWidget.setCurrentWidget(widget, popOut=False)

    def closeEvent(self, event):
        self.windowHidden.emit(True)
        super().closeEvent(event)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    font = QFont("Arial", 11)
    app.setFont(font)

    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
