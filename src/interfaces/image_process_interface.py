import os
import sys
import csv
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QCheckBox,
    QDoubleSpinBox,
)
import tqdm
from PIL import Image
from PyQt5.QtWidgets import QSlider
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from src.gpu_utils.gpu_image_process import (
    GPUImageProcess,
    gpu_convert_raw_to_tiff,
    gpu_crop_image,
    gpu_rotate_image,
    gpu_sharpen_edge,
    gpu_negative_log,
    gpu_rotate_r90,
)
from src.gpu_utils.gpu_batch_operations import GPUBatchOperations


def reset_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            sub_layout = item.layout()
            if sub_layout is not None:
                reset_layout(sub_layout)


class IconTextButton(QPushButton):
    def __init__(self, text, icon_path, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        icon_label = QLabel(self)
        pixmap = QPixmap(icon_path)
        scaled_pixmap = pixmap.scaled(
            32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(32, 32)
        text_label = QLabel(text, self)
        text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        self.setLayout(layout)


class SetImageSizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Image Size")
        self.layout = QVBoxLayout()
        self.width_label = QLabel("Image Width:")
        self.width_input = QLineEdit(self)
        self.width_input.setText("2340")
        self.layout.addWidget(self.width_label)
        self.layout.addWidget(self.width_input)
        self.height_label = QLabel("Image Height:")
        self.height_input = QLineEdit(self)
        self.height_input.setText("2882")
        self.layout.addWidget(self.height_label)
        self.layout.addWidget(self.height_input)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def get_size(self):
        width = int(self.width_input.text())
        height = int(self.height_input.text())
        return width, height


class RotateAngleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rotate Angle")
        self.layout = QVBoxLayout()
        self.angle_label = QLabel("逆时针旋转角度:")
        self.angle_input = QLineEdit(self)
        self.layout.addWidget(self.angle_label)
        self.layout.addWidget(self.angle_input)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def get_angle(self):
        return float(self.angle_input.text())


class CropCoordinatesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.layout = QVBoxLayout()
        self.x1_label = QLabel("Top left X:")
        self.x1_input = QLineEdit(self)
        self.layout.addWidget(self.x1_label)
        self.layout.addWidget(self.x1_input)
        self.y1_label = QLabel("Top left Y:")
        self.y1_input = QLineEdit(self)
        self.layout.addWidget(self.y1_label)
        self.layout.addWidget(self.y1_input)
        self.x2_label = QLabel("Bottom right X:")
        self.x2_input = QLineEdit(self)
        self.layout.addWidget(self.x2_label)
        self.layout.addWidget(self.x2_input)
        self.y2_label = QLabel("Bottom right Y:")
        self.y2_input = QLineEdit(self)
        self.layout.addWidget(self.y2_label)
        self.layout.addWidget(self.y2_input)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def get_crop_coordinates(self):
        x1 = int(self.x1_input.text())
        y1 = int(self.y1_input.text())
        x2 = int(self.x2_input.text())
        y2 = int(self.y2_input.text())
        return (x1, y1, x2, y2)


class ImageInfoDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像信息")
        self.image_path = image_path
        self.image_data = None
        self.load_image()
        self.setup_ui()
        self.setFixedSize(900, 600)

    def load_image(self):
        if self.image_path.endswith('.raw'):
            raw_data = np.fromfile(self.image_path, dtype=np.uint16)
            self.image_data = raw_data.reshape((2882, 2340))
        else:
            from PIL import Image as PILImage
            img = PILImage.open(self.image_path)
            self.image_data = np.array(img)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        image_info_layout = QVBoxLayout()
        height, width = self.image_data.shape
        min_val = int(np.min(self.image_data))
        max_val = int(np.max(self.image_data))
        mean_val = float(np.mean(self.image_data))
        std_val = float(np.std(self.image_data))

        info_text = f"""
        <h3>图像信息</h3>
        <p><b>文件路径:</b> {self.image_path}</p>
        <p><b>图像尺寸:</b> {width} x {height}</p>
        <p><b>像素范围:</b> {min_val} - {max_val}</p>
        <p><b>像素均值:</b> {mean_val:.2f}</p>
        <p><b>像素标准差:</b> {std_val:.2f}</p>
        """
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-size: 12px;")
        image_info_layout.addWidget(info_label)
        layout.addLayout(image_info_layout)

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.plot_histogram()

    def plot_histogram(self):
        ax = self.figure.add_subplot(111)
        
        hist, bins, patches = ax.hist(self.image_data.flatten(), bins=100, edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('像素值')
        ax.set_ylabel('数量')
        ax.set_title('像素值分布直方图')
        ax.grid(True, alpha=0.3)

        max_count = np.max(hist)
        threshold = max_count * 0.1

        for i, count in enumerate(hist):
            if count < threshold:
                x_pos = bins[i] + (bins[1] - bins[0]) / 2
                ax.text(x_pos, count, str(int(count)), 
                       ha='center', va='bottom', fontsize=8, rotation=90)

        self.figure.tight_layout()
        self.canvas.draw()


class ThresholdDialog(QDialog):
    def __init__(self, min_val, max_val, parent=None):
        super().__init__(parent)
        self.setWindowTitle("阈值划分设置")
        self.min_val = min_val
        self.max_val = max_val
        self.setup_ui()
        self.setFixedSize(400, 350)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("选择模式:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)

        self.mode_group = QButtonGroup()
        self.percent_radio = QRadioButton("百分比")
        self.percent_radio.setChecked(True)
        self.mode_group.addButton(self.percent_radio)
        mode_layout.addWidget(self.percent_radio)

        self.value_radio = QRadioButton("真实值")
        self.mode_group.addButton(self.value_radio)
        mode_layout.addWidget(self.value_radio)

        mode_layout.addStretch(1)
        layout.addLayout(mode_layout)

        self.percent_layout = QHBoxLayout()
        min_percent_label = QLabel("最小百分比 (%):")
        self.min_percent_input = QLineEdit()
        self.min_percent_input.setText("10")
        self.percent_layout.addWidget(min_percent_label)
        self.percent_layout.addWidget(self.min_percent_input)
        layout.addLayout(self.percent_layout)

        max_percent_layout = QHBoxLayout()
        max_percent_label = QLabel("最大百分比 (%):")
        self.max_percent_input = QLineEdit()
        self.max_percent_input.setText("80")
        max_percent_layout.addWidget(max_percent_label)
        max_percent_layout.addWidget(self.max_percent_input)
        layout.addLayout(max_percent_layout)

        self.value_layout = QHBoxLayout()
        min_value_label = QLabel("最小值:")
        self.min_value_input = QLineEdit()
        self.min_value_input.setText(str(self.min_val))
        self.value_layout.addWidget(min_value_label)
        self.value_layout.addWidget(self.min_value_input)
        layout.addLayout(self.value_layout)

        max_value_layout = QHBoxLayout()
        max_value_label = QLabel("最大值:")
        self.max_value_input = QLineEdit()
        self.max_value_input.setText(str(self.max_val))
        max_value_layout.addWidget(max_value_label)
        max_value_layout.addWidget(self.max_value_input)
        layout.addLayout(max_value_layout)

        radius_layout = QHBoxLayout()
        radius_label = QLabel("滤波半径:")
        radius_label.setStyleSheet("font-weight: bold;")
        radius_layout.addWidget(radius_label)

        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setRange(1, 10)
        self.radius_spinbox.setValue(2)
        radius_layout.addWidget(self.radius_spinbox)
        radius_layout.addStretch(1)
        layout.addLayout(radius_layout)

        self.percent_radio.toggled.connect(self.on_mode_changed)
        self.value_radio.toggled.connect(self.on_mode_changed)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        layout.addStretch(1)

        self.on_mode_changed()

    def on_mode_changed(self):
        is_percent = self.percent_radio.isChecked()
        
        for i in range(self.percent_layout.count()):
            widget = self.percent_layout.itemAt(i).widget()
            if widget:
                widget.setEnabled(is_percent)

        for i in range(self.value_layout.count()):
            widget = self.value_layout.itemAt(i).widget()
            if widget:
                widget.setEnabled(not is_percent)

    def get_threshold_params(self):
        mode = "percent" if self.percent_radio.isChecked() else "value"
        radius = self.radius_spinbox.value()
        
        if mode == "percent":
            min_percent = float(self.min_percent_input.text())
            max_percent = float(self.max_percent_input.text())
            return mode, min_percent, max_percent, radius
        else:
            min_value = float(self.min_value_input.text())
            max_value = float(self.max_value_input.text())
            return mode, min_value, max_value, radius


class BinarizeDialog(QDialog):
    def __init__(self, min_val, max_val, parent=None):
        super().__init__(parent)
        self.setWindowTitle("二值化设置")
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        if self.min_val == self.max_val:
            self.max_val = self.min_val + 1.0
        self.setup_ui()
        self.setFixedSize(420, 200)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("阈值:")
        threshold_label.setStyleSheet("font-weight: bold;")
        threshold_layout.addWidget(threshold_label)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(self.min_val, self.max_val)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue((self.min_val + self.max_val) / 2.0)
        self.threshold_spin.setSingleStep(1.0)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch(1)
        layout.addLayout(threshold_layout)

        self.invert_checkbox = QCheckBox("0/255 反转（阈值以上设0，以下设255）")
        layout.addWidget(self.invert_checkbox)

        hint_label = QLabel("规则：阈值以上(>=)输出255，以下输出0")
        hint_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint_label)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_params(self):
        return self.threshold_spin.value(), self.invert_checkbox.isChecked()


class SmallRegionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("小区域删除设置")
        self.setup_ui()
        self.setFixedSize(420, 180)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        pixel_layout = QHBoxLayout()
        pixel_label = QLabel("最小连通域像素数:")
        pixel_label.setStyleSheet("font-weight: bold;")
        pixel_layout.addWidget(pixel_label)

        self.pixel_spin = QSpinBox()
        self.pixel_spin.setRange(1, 100000000)
        self.pixel_spin.setValue(20)
        pixel_layout.addWidget(self.pixel_spin)
        pixel_layout.addStretch(1)
        layout.addLayout(pixel_layout)

        hint_label = QLabel("小于该像素数的孤岛连通域会被置为0")
        hint_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint_label)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_min_pixels(self):
        return self.pixel_spin.value()


class ImageProcessInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ImageProcessInterface")
        self.setWindowTitle("GPU Image Process")
        self.setGeometry(500, 500, 1400, 800)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)

        self.model = QFileSystemModel()
        self.model.setRootPath("")

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(""))
        self.tree_view.clicked.connect(self.on_tree_view_clicked)
        for i in range(1, self.model.columnCount()):
            self.tree_view.setColumnHidden(i, True)
        self.tree_view.setMinimumWidth(400)

        self.dock_widget = QDockWidget("File System", self)
        self.dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.dock_widget.setMinimumWidth(400)

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)

        self.select_input_button = IconTextButton(
            "选定输入", "./resource/icons/data_input.png", self
        )
        self.select_input_button.setFixedSize(150, 70)
        self.select_input_button.clicked.connect(self.on_select_input_clicked)

        self.select_output_button = IconTextButton(
            "选定输出", "./resource/icons/data_output.png", self
        )
        self.select_output_button.setFixedSize(150, 70)
        self.select_output_button.clicked.connect(self.on_select_output_clicked)

        self.cancel_button = IconTextButton("Clear", "./resource/icons/clear.png", self)
        self.cancel_button.setFixedSize(150, 70)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

        self.set_image_size_button = IconTextButton(
            "设置尺寸", "./resource/icons/size.png", self
        )
        self.set_image_size_button.setFixedSize(150, 70)
        self.set_image_size_button.clicked.connect(self.on_set_image_size_clicked)

        self.convert_button = IconTextButton(
            "Raw转Tiff", "./resource/icons/convert_dtype.png", self
        )
        self.convert_button.setFixedSize(150, 70)
        self.convert_button.clicked.connect(self.on_convert_clicked)

        self.crop_button = IconTextButton(
            "裁剪图像", "./resource/icons/crop.png", self
        )
        self.crop_button.setFixedSize(150, 70)
        self.crop_button.clicked.connect(self.on_crop_clicked)

        self.rotate_r90_button = IconTextButton(
            "顺时针90°", "./resource/icons/R_spin.png", self
        )
        self.rotate_r90_button.setFixedSize(150, 70)
        self.rotate_r90_button.clicked.connect(self.on_rotate_r90_clicked)

        self.rotate_any_button = IconTextButton(
            "逆时针任意", "./resource/icons/L_spin.png", self
        )
        self.rotate_any_button.setFixedSize(150, 70)
        self.rotate_any_button.clicked.connect(self.on_rotate_any_clicked)

        self.sharpen_button = IconTextButton(
            "边缘锐化", "./resource/icons/sharpen.png", self
        )
        self.sharpen_button.setFixedSize(150, 70)
        self.sharpen_button.clicked.connect(self.on_sharpen_clicked)

        self.negative_log_button = IconTextButton(
            "负对数", "./resource/icons/log.png", self
        )
        self.negative_log_button.setFixedSize(150, 70)
        self.negative_log_button.clicked.connect(self.on_negative_log_clicked)

        self.info_button = IconTextButton(
            "Info", "./resource/icons/info.png", self
        )
        self.info_button.setFixedSize(150, 70)
        self.info_button.clicked.connect(self.on_info_clicked)

        self.threshold_button = IconTextButton(
            "阈值划分", "./resource/icons/threshold.png", self
        )
        self.threshold_button.setFixedSize(150, 70)
        self.threshold_button.clicked.connect(self.on_threshold_clicked)

        self.binarize_button = IconTextButton(
            "图像二值化", "./resource/icons/check.png", self
        )
        self.binarize_button.setFixedSize(150, 70)
        self.binarize_button.clicked.connect(self.on_binarize_clicked)

        self.remove_small_region_button = IconTextButton(
            "小区域删除", "./resource/icons/change.png", self
        )
        self.remove_small_region_button.setFixedSize(150, 70)
        self.remove_small_region_button.clicked.connect(self.on_remove_small_region_clicked)

        self.extract_transition_button = IconTextButton(
            "突变点提取", "./resource/icons/progress.png", self
        )
        self.extract_transition_button.setFixedSize(150, 70)
        self.extract_transition_button.clicked.connect(self.on_extract_transition_clicked)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.select_input_button)
        button_layout.addWidget(self.select_output_button)
        button_layout.addWidget(self.set_image_size_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.convert_button)
        button_layout.addWidget(self.crop_button)
        button_layout.addWidget(self.rotate_r90_button)
        button_layout.addWidget(self.rotate_any_button)
        button_layout.addWidget(self.sharpen_button)
        button_layout.addWidget(self.negative_log_button)
        button_layout.addWidget(self.info_button)
        button_layout.addWidget(self.threshold_button)
        button_layout.addWidget(self.binarize_button)
        button_layout.addWidget(self.remove_small_region_button)
        button_layout.addWidget(self.extract_transition_button)
        button_layout.addStretch(1)

        mid_central_layout = QVBoxLayout()
        self.parameter_window = QVBoxLayout()
        mid_central_layout.addWidget(self.text)
        button_layout.addLayout(self.parameter_window)

        central_layout.addWidget(self.tree_view)
        central_layout.addLayout(mid_central_layout)
        central_layout.addLayout(button_layout)

        self.setCentralWidget(central_widget)
        self.dock_widget.setWidget(self.tree_view)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget)

        self.current_path = None
        self.input_path = None
        self.output_path = None
        self.image_width = 2340
        self.image_height = 2882
        self.batch_size = 100
        self.batch_operations = GPUBatchOperations(batch_size=self.batch_size)

    def reset(self):
        reset_layout(self.parameter_window)

    def createInputLayout(self, label_text):
        layout = QVBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit(self)
        input_field.setFixedSize(150, 35)
        layout.addWidget(label)
        layout.addWidget(input_field)
        return layout

    def _collect_image_paths(self):
        if not self.input_path or not os.path.isdir(self.input_path):
            return []

        valid_exts = (".tif", ".tiff", ".raw")
        return [
            os.path.join(self.input_path, img)
            for img in os.listdir(self.input_path)
            if img.lower().endswith(valid_exts)
        ]

    def _load_image_data(self, image_path):
        if image_path.lower().endswith(".raw"):
            raw_data = np.fromfile(image_path, dtype=np.uint16)
            expected_size = self.image_width * self.image_height
            if raw_data.size != expected_size:
                raise ValueError(
                    f"RAW尺寸不匹配: 当前设置为 {self.image_width}x{self.image_height} "
                    f"(共{expected_size}像素), 实际文件像素数 {raw_data.size}"
                )
            return raw_data.reshape((self.image_height, self.image_width))

        image = Image.open(image_path)
        img_data = np.array(image)
        if img_data.ndim == 3:
            img_data = img_data[:, :, 0]
        return img_data

    def _build_output_path(self, image_path):
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        if ext.lower() == ".raw":
            return os.path.join(self.output_path, f"{name}.tif")
        return os.path.join(self.output_path, basename)

    def _remove_small_regions(self, binary_img, min_pixels):
        from scipy import ndimage

        binary_01 = (binary_img > 0).astype(np.uint8)
        structure = np.ones((3, 3), dtype=np.uint8)
        labeled, num_features = ndimage.label(binary_01, structure=structure)

        if num_features == 0:
            return (binary_01 * 255).astype(np.uint8), 0

        component_sizes = np.bincount(labeled.ravel())
        small_components = component_sizes < min_pixels
        small_components[0] = False

        removed_count = int(np.sum(small_components[1:]))
        binary_01[small_components[labeled]] = 0
        return (binary_01 * 255).astype(np.uint8), removed_count

    def on_tree_view_clicked(self, index):
        self.current_path = self.model.filePath(index)

    def on_select_input_clicked(self):
        if self.current_path:
            self.text.append(f"Input Path: {self.current_path}\n")
            self.input_path = self.current_path
            self.current_path = None
        else:
            QMessageBox.warning(self, "No Selection", "No folder selected.")

    def on_select_output_clicked(self):
        if self.current_path:
            self.text.append(f"Output Path: {self.current_path}\n")
            self.output_path = self.current_path
            self.current_path = None
        else:
            QMessageBox.warning(self, "No Selection", "No folder selected.")

    def on_cancel_clicked(self):
        self.input_path = None
        self.output_path = None
        QMessageBox.information(self, "Cleared", "Selection cleared.")
        self.text.append(f"ALL Path: None ")
        self.reset()
        self.text.append(f"-----------------------------------------------------------")

    def on_set_image_size_clicked(self):
        dialog = SetImageSizeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            width, height = dialog.get_size()
            if width and height:
                self.image_width = width
                self.image_height = height
                self.text.append(f"Set image size to: {self.image_width} x {self.image_height}")

    def on_convert_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select input and output folders.")
            return

        image_paths = [
            os.path.join(self.input_path, img)
            for img in os.listdir(self.input_path)
            if img.endswith(".raw")
        ]

        for image in tqdm.tqdm(image_paths, desc="Converting"):
            try:
                gpu_convert_raw_to_tiff(image, self.output_path, self.image_width, self.image_height)
                self.text.append(
                    f'<span style="color: green;"> Successfully converted {image}</span>'
                )
            except Exception as e:
                self.text.append(
                    f'<span style="color: red;">Error processing {image}: {str(e)} </span>'
                )

        self.text.append(f"\n")

    def on_crop_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select input and output folders.")
            return

        dialog = CropCoordinatesDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            x1, y1, x2, y2 = dialog.get_crop_coordinates()

            image_paths = [
                os.path.join(self.input_path, img)
                for img in os.listdir(self.input_path)
                if img.endswith(".tif") or img.endswith(".tiff")
            ]

            for image in tqdm.tqdm(image_paths, desc="Cropping"):
                try:
                    gpu_crop_image(image, self.output_path, x1, y1, x2, y2)
                    self.text.append(
                        f'<span style="color: green;"> Successfully cropped {image}</span>'
                    )
                except Exception as e:
                    self.text.append(
                        f'<span style="color: red;">Error processing {image}: {str(e)} </span>'
                    )
        self.text.append(f"\n")

    def on_rotate_r90_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select input and output folders.")
            return

        image_paths = [
            os.path.join(self.input_path, img)
            for img in os.listdir(self.input_path)
            if img.endswith((".tif", ".tiff"))
        ]

        def progress_callback(processed, total):
            if processed % 10 == 0:
                self.text.append(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")

        self.batch_operations.set_progress_callback(progress_callback)
        self.batch_operations.batch_rotate_r90(image_paths, self.output_path)
        self.text.append(f'<span style="color: green;">旋转处理完成！</span>')
        self.text.append(f"\n")

    def on_rotate_any_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select input and output folders.")
            return

        dialog = RotateAngleDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            angle = dialog.get_angle()

            image_paths = [
                os.path.join(self.input_path, img)
                for img in os.listdir(self.input_path)
                if img.endswith(".tif") or img.endswith(".tiff")
            ]

            for image in tqdm.tqdm(image_paths, desc="Rotating"):
                try:
                    gpu_rotate_image(image, self.output_path, angle)
                    self.text.append(
                        f'<span style="color: green;">Successfully rotated {image}</span>'
                    )
                except Exception as e:
                    self.text.append(
                        f'<span style="color: red;">Error processing {image}: {str(e)}</span>'
                    )
        self.text.append(f"\n")

    def on_sharpen_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select input and output folders.")
            return

        image_paths = [
            os.path.join(self.input_path, img)
            for img in os.listdir(self.input_path)
            if img.endswith(".tif") or img.endswith(".tiff")
        ]

        for image in tqdm.tqdm(image_paths, desc="Sharpening"):
            try:
                gpu_sharpen_edge(image, self.output_path)
                self.text.append(
                    f'<span style="color: green;"> Successfully sharpened {image}</span>'
                )
            except Exception as e:
                self.text.append(
                    f'<span style="color: red;">Error processing {image}: {str(e)} </span>'
                )

        self.text.append(f"\n")

    def on_negative_log_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select input and output folders.")
            return

        image_paths = [
            os.path.join(self.input_path, img)
            for img in os.listdir(self.input_path)
            if img.endswith((".tif", ".tiff"))
        ]

        def progress_callback(processed, total):
            if processed % 10 == 0:
                self.text.append(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")

        self.batch_operations.set_progress_callback(progress_callback)
        self.batch_operations.batch_negative_log(image_paths, self.output_path)
        self.text.append(f'<span style="color: green;">取负对数处理完成！</span>')
        self.text.append(f"\n")

    def on_info_clicked(self):
        if not self.current_path:
            QMessageBox.warning(self, "警告", "请先选择一个图像文件！")
            return

        if os.path.isdir(self.current_path):
            QMessageBox.warning(self, "警告", "请选择图像文件而不是文件夹！")
            return

        if not (self.current_path.endswith((".tif", ".tiff", ".raw"))):
            QMessageBox.warning(self, "警告", "只支持 .tif, .tiff 或 .raw 格式的图像文件！")
            return

        dialog = ImageInfoDialog(self.current_path, self)
        dialog.exec_()

        image_info = dialog.image_data
        height, width = image_info.shape
        min_val = int(np.min(image_info))
        max_val = int(np.max(image_info))
        mean_val = float(np.mean(image_info))
        std_val = float(np.std(image_info))

        self.text.append(f"-----------------------------------------------------------")
        self.text.append(f"<b>图像信息:</b> {self.current_path}")
        self.text.append(f"  图像尺寸: {width} x {height}")
        self.text.append(f"  像素范围: {min_val} - {max_val}")
        self.text.append(f"  像素均值: {mean_val:.2f}")
        self.text.append(f"  像素标准差: {std_val:.2f}")
        self.text.append(f"-----------------------------------------------------------")

    def on_threshold_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        image_paths = [
            os.path.join(self.input_path, img)
            for img in os.listdir(self.input_path)
            if img.endswith((".tif", ".tiff", ".raw"))
        ]

        if not image_paths:
            QMessageBox.warning(self, "警告", "输入文件夹中没有找到图像文件！")
            return

        first_image_path = image_paths[0]
        if first_image_path.endswith('.raw'):
            raw_data = np.fromfile(first_image_path, dtype=np.uint16)
            sample_image = raw_data.reshape((2882, 2340))
        else:
            from PIL import Image as PILImage
            img = PILImage.open(first_image_path)
            sample_image = np.array(img)

        min_val = int(np.min(sample_image))
        max_val = int(np.max(sample_image))

        dialog = ThresholdDialog(min_val, max_val, self)
        if dialog.exec_() == QDialog.Accepted:
            mode, param1, param2, radius = dialog.get_threshold_params()

            from scipy.ndimage import median_filter

            for image_path in tqdm.tqdm(image_paths, desc="Thresholding"):
                try:
                    if image_path.endswith('.raw'):
                        raw_data = np.fromfile(image_path, dtype=np.uint16)
                        img_data = raw_data.reshape((2882, 2340))
                    else:
                        from PIL import Image as PILImage
                        img = PILImage.open(image_path)
                        img_data = np.array(img)

                    if mode == "percent":
                        min_percent = param1
                        max_percent = param2
                        
                        flat_pixels = img_data.flatten()
                        sorted_pixels = np.sort(flat_pixels)
                        total_pixels = len(flat_pixels)
                        
                        low_count = int(total_pixels * min_percent / 100)
                        high_count = int(total_pixels * (100 - max_percent) / 100)
                        
                        threshold_min = sorted_pixels[low_count]
                        threshold_max = sorted_pixels[total_pixels - high_count - 1]
                    else:
                        threshold_min = param1
                        threshold_max = param2

                    result_img = img_data.copy().astype(np.float32)
                    
                    mask_valid = (img_data >= threshold_min) & (img_data <= threshold_max)
                    mask_invalid = ~mask_valid
                    
                    if np.any(mask_invalid):
                        filtered = median_filter(img_data, size=radius * 2 + 1)
                        result_img[mask_invalid] = filtered[mask_invalid]

                    basename = os.path.basename(image_path)
                    output_file = os.path.join(self.output_path, basename)
                    
                    from PIL import Image as PILImage
                    output_img = PILImage.fromarray(result_img.astype(np.float32))
                    output_img.save(output_file)

                    self.text.append(
                        f'<span style="color: green;">成功处理 {image_path}, 阈值范围: {threshold_min:.1f} - {threshold_max:.1f}, 滤波半径: {radius}</span>'
                    )
                except Exception as e:
                    self.text.append(
                        f'<span style="color: red;">处理 {image_path} 失败: {str(e)}</span>'
                    )

            self.text.append(f"\n")
            self.text.append(f'<span style="color: green;">阈值划分处理完成！</span>')

    def on_binarize_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        image_paths = self._collect_image_paths()
        if not image_paths:
            QMessageBox.warning(self, "警告", "输入文件夹中没有找到 .tif/.tiff/.raw 图像！")
            return

        try:
            sample_img = self._load_image_data(image_paths[0])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取样本图像失败: {e}")
            return

        min_val = float(np.min(sample_img))
        max_val = float(np.max(sample_img))
        dialog = BinarizeDialog(min_val, max_val, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        threshold, invert = dialog.get_params()
        os.makedirs(self.output_path, exist_ok=True)

        for image_path in tqdm.tqdm(image_paths, desc="Binarizing"):
            try:
                img_data = self._load_image_data(image_path)
                binary_img = np.where(img_data >= threshold, 255, 0).astype(np.uint8)
                if invert:
                    binary_img = 255 - binary_img

                output_path = self._build_output_path(image_path)
                Image.fromarray(binary_img).save(output_path)

                self.text.append(
                    f'<span style="color: green;">二值化成功: {os.path.basename(image_path)} -> {os.path.basename(output_path)}</span>'
                )
            except Exception as e:
                self.text.append(
                    f'<span style="color: red;">二值化失败: {image_path}, 原因: {e}</span>'
                )

        self.text.append(
            f'<span style="color: green;">二值化处理完成（阈值={threshold:.3f}, 反转={"是" if invert else "否"}）</span>'
        )
        self.text.append("\n")

    def on_remove_small_region_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        image_paths = self._collect_image_paths()
        if not image_paths:
            QMessageBox.warning(self, "警告", "输入文件夹中没有找到 .tif/.tiff/.raw 图像！")
            return

        dialog = SmallRegionDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        min_pixels = dialog.get_min_pixels()
        os.makedirs(self.output_path, exist_ok=True)

        for image_path in tqdm.tqdm(image_paths, desc="Removing Small Regions"):
            try:
                img_data = self._load_image_data(image_path)
                binary_img = np.where(img_data > 0, 255, 0).astype(np.uint8)
                cleaned_img, removed_count = self._remove_small_regions(binary_img, min_pixels)

                output_path = self._build_output_path(image_path)
                Image.fromarray(cleaned_img).save(output_path)

                self.text.append(
                    f'<span style="color: green;">小区域删除成功: {os.path.basename(image_path)} '
                    f'(移除连通域 {removed_count} 个)</span>'
                )
            except Exception as e:
                self.text.append(
                    f'<span style="color: red;">小区域删除失败: {image_path}, 原因: {e}</span>'
                )

        self.text.append(
            f'<span style="color: green;">小区域删除处理完成（最小像素阈值={min_pixels}）</span>'
        )
        self.text.append("\n")

    def on_extract_transition_clicked(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        image_paths = self._collect_image_paths()
        if not image_paths:
            QMessageBox.warning(self, "警告", "输入文件夹中没有找到 .tif/.tiff/.raw 图像！")
            return

        os.makedirs(self.output_path, exist_ok=True)
        success_count = 0
        fail_count = 0

        for image_path in tqdm.tqdm(image_paths, desc="Extracting Transition Points"):
            try:
                img_data = self._load_image_data(image_path)
                binary_img = np.where(img_data > 0, 255, 0).astype(np.uint8)
                transition_points = []

                for row_index in range(binary_img.shape[0]):
                    row = binary_img[row_index]
                    if row.size < 2:
                        transition_points.append((row_index + 1, None))
                        continue

                    transitions = np.where((row[:-1] == 0) & (row[1:] == 255))[0]
                    if transitions.size > 0:
                        first_col = int(transitions[0]) + 2
                        transition_points.append((row_index + 1, first_col))
                    else:
                        transition_points.append((row_index + 1, None))

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                txt_save_path = os.path.join(
                    self.output_path, f"{base_name}_transition_points.txt"
                )
                csv_save_path = os.path.join(
                    self.output_path, f"{base_name}_transition_points.csv"
                )

                with open(txt_save_path, "w", encoding="utf-8") as f:
                    for row_x, col_y in transition_points:
                        y_text = "Nan" if col_y is None else str(col_y)
                        f.write(f"第{row_x}行 突变点为第{y_text}列\n")

                with open(csv_save_path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["X", "Y"])
                    for row_x, col_y in transition_points:
                        writer.writerow([row_x, col_y if col_y is not None else "nan"])

                valid_count = sum(1 for _, y in transition_points if y is not None)
                self.text.append(
                    f'<span style="color: green;">突变点提取成功: {os.path.basename(image_path)} '
                    f'(总行数: {len(transition_points)}, 有效行: {valid_count})</span>'
                )
                self.text.append(f"  结果文件: {txt_save_path}")
                self.text.append(f"  结果文件: {csv_save_path}")
                success_count += 1
            except Exception as e:
                self.text.append(
                    f'<span style="color: red;">突变点提取失败: {image_path}, 原因: {e}</span>'
                )
                fail_count += 1

        self.text.append("-----------------------------------------------------------")
        self.text.append(
            f"<b>批量突变点提取完成</b> 成功: {success_count}, 失败: {fail_count}"
        )
        self.text.append(f"输出目录: {self.output_path}")
        self.text.append("-----------------------------------------------------------")
        self.text.append("\n")
