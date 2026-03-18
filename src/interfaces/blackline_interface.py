import os
import sys
import time

import cupy as cp
import tqdm
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFileSystemModel,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from src.gpu_utils.gpu_batch_processor import GPUBatchProcessor
from src.gpu_utils.gpu_image_process import GPUImageProcess
from src.interfaces.ui_theme import apply_interface_theme, set_button_role


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


class BlacklineSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)

        title_label = QLabel("黑线检测参数设置")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(10)

        self.stacked_widget = QStackedWidget()

        page1 = self.create_txt_file_page()
        page2 = self.create_auto_detect_page()

        self.stacked_widget.addWidget(page1)
        self.stacked_widget.addWidget(page2)

        self.main_layout.addWidget(self.stacked_widget)

        button_layout = QHBoxLayout()
        self.txt_file_button = QPushButton("从文件读取")
        self.txt_file_button.setCheckable(True)
        self.txt_file_button.setChecked(True)
        self.txt_file_button.clicked.connect(lambda: self.switch_page(0))

        self.auto_detect_button = QPushButton("自动检测")
        self.auto_detect_button.setCheckable(True)
        self.auto_detect_button.clicked.connect(lambda: self.switch_page(1))

        button_layout.addWidget(self.txt_file_button)
        button_layout.addWidget(self.auto_detect_button)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addSpacing(10)

        self.confirm_button = QPushButton("确认设置")
        set_button_role(self.confirm_button, "primary")
        self.confirm_button.clicked.connect(self.confirm_settings)
        self.main_layout.addWidget(self.confirm_button)

        self.main_layout.addStretch(1)

        self.current_mode = "txt_file"
        self.columns = None
        self.min_val = 2000
        self.max_val = 8000
        self.grad = 4.0
        self.blackline_lp = 2

    def create_txt_file_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        info_label = QLabel("从 resource/blacklines_test/ 目录读取txt文件")
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        layout.addSpacing(10)

        self.txt_file_list = QListWidget()
        layout.addWidget(self.txt_file_list)
        layout.addSpacing(10)

        self.refresh_txt_files()

        refresh_button = QPushButton("刷新文件列表")
        refresh_button.clicked.connect(self.refresh_txt_files)
        layout.addWidget(refresh_button)

        return page

    def create_auto_detect_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.min_val_label = QLabel("最小值:")
        self.min_val_input = QLineEdit(self)
        self.min_val_input.setText("2000")
        layout.addWidget(self.min_val_label)
        layout.addWidget(self.min_val_input)

        self.max_val_label = QLabel("最大值:")
        self.max_val_input = QLineEdit(self)
        self.max_val_input.setText("8000")
        layout.addWidget(self.max_val_label)
        layout.addWidget(self.max_val_input)

        self.blackline_lp_label = QLabel("黑线参数 blackline_lp:")
        self.blackline_lp_input = QLineEdit(self)
        self.blackline_lp_input.setText("2")
        layout.addWidget(self.blackline_lp_label)
        layout.addWidget(self.blackline_lp_input)

        self.grad_label = QLabel("允许梯度:")
        self.grad_input = QLineEdit(self)
        self.grad_input.setText("4.0")
        layout.addWidget(self.grad_label)
        layout.addWidget(self.grad_input)

        layout.addStretch(1)

        return page

    def switch_page(self, index):
        if index == 0:
            self.txt_file_button.setChecked(True)
            self.auto_detect_button.setChecked(False)
        else:
            self.txt_file_button.setChecked(False)
            self.auto_detect_button.setChecked(True)

        self.stacked_widget.setCurrentIndex(index)
        self.current_mode = "txt_file" if index == 0 else "auto_detect"

    def refresh_txt_files(self):
        self.txt_file_list.clear()
        blacklines_dir = "resource/blacklines_test"
        if os.path.exists(blacklines_dir):
            txt_files = [f for f in os.listdir(blacklines_dir) if f.endswith(".txt")]
            self.txt_file_list.addItems(txt_files)

    def confirm_settings(self):
        if self.current_mode == "txt_file":
            selected_item = self.txt_file_list.currentItem()
            if not selected_item:
                QMessageBox.warning(self, "警告", "请先选择一个txt文件！")
                return

            blacklines_dir = "resource/blacklines_test"
            selected_file = selected_item.text()
            file_path = os.path.join(blacklines_dir, selected_file)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = [
                        int(line.strip()) for line in file if line.strip().isdigit()
                    ]
                self.columns = data
                self.current_mode = "txt_file"
                self.min_val = 2000
                self.max_val = 8000
                self.grad = 4.0
                self.blackline_lp = 2
                QMessageBox.information(
                    self,
                    "成功",
                    f"从文件 {selected_file} 读取到 {len(data)} 条坏线\n坏线列: {data[:10]}{'...' if len(data) > 10 else ''}",
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取文件失败: {e}")
        else:
            try:
                self.min_val = int(self.min_val_input.text())
                self.max_val = int(self.max_val_input.text())
                self.grad = float(self.grad_input.text())
                self.blackline_lp = int(self.blackline_lp_input.text())
                self.columns = None
                self.current_mode = "auto_detect"
                QMessageBox.information(
                    self,
                    "成功",
                    f"参数设置成功！\n最小值: {self.min_val}, 最大值: {self.max_val}\n梯度阈值: {self.grad}, blackline_lp: {self.blackline_lp}",
                )
            except ValueError as e:
                QMessageBox.critical(self, "错误", f"参数格式错误: {e}")

    def get_settings(self):
        if self.current_mode == "txt_file":
            selected_item = self.txt_file_list.currentItem()
            if selected_item:
                blacklines_dir = "resource/blacklines_test"
                selected_file = selected_item.text()
                file_path = os.path.join(blacklines_dir, selected_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = [
                            int(line.strip()) for line in file if line.strip().isdigit()
                        ]
                    self.columns = data
                except Exception:
                    pass

            return {
                "mode": self.current_mode,
                "columns": self.columns,
                "min_val": 2000,
                "max_val": 8000,
                "grad": 4.0,
                "blackline_lp": 2,
            }
        else:
            try:
                min_val = int(self.min_val_input.text())
                max_val = int(self.max_val_input.text())
                grad = float(self.grad_input.text())
                blackline_lp = int(self.blackline_lp_input.text())

                return {
                    "mode": self.current_mode,
                    "columns": None,
                    "min_val": min_val,
                    "max_val": max_val,
                    "grad": grad,
                    "blackline_lp": blackline_lp,
                }
            except ValueError:
                return {
                    "mode": self.current_mode,
                    "columns": None,
                    "min_val": 2000,
                    "max_val": 8000,
                    "grad": 4.0,
                    "blackline_lp": 2,
                }


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


class BlacklineInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("BlacklineInterface")
        self.setWindowTitle("GPU黑线去除")
        self.setGeometry(500, 500, 1400, 800)
        apply_interface_theme(self)

        self.gpu_memory_usage = 0.95
        self.cuda_stream_count = 2
        self.batch_size = 100

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

        self.dock_widget = QDockWidget("文件系统", self)
        self.dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.dock_widget.setMinimumWidth(400)

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)

        self.select_input_button = IconTextButton(
            "选定输入", "./resource/icons/data_input.png", self
        )
        self.select_input_button.setFixedSize(200, 70)
        self.select_input_button.clicked.connect(self.on_select_input_clicked)

        self.select_output_button = IconTextButton(
            "选定输出", "./resource/icons/data_output.png", self
        )
        self.select_output_button.setFixedSize(200, 70)
        self.select_output_button.clicked.connect(self.on_select_output_clicked)

        self.cancel_button = IconTextButton(
            "清空选择", "./resource/icons/clear.png", self
        )
        self.cancel_button.setFixedSize(200, 70)
        set_button_role(self.cancel_button, "danger")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

        self.process_method_label = QLabel("选择背底处理方法:")
        self.process_method_combo = QComboBox(self)
        self.process_method_combo.setFixedSize(200, 35)

        self.process_method_combo.addItems(["单张白减黑", "多张白减黑均值"])

        self.process_background_button = IconTextButton(
            "处理背底", "./resource/icons/back.png", self
        )
        self.process_background_button.setFixedSize(200, 70)
        set_button_role(self.process_background_button, "primary")
        self.process_background_button.clicked.connect(self.process_background)

        self.process_tomos_button = IconTextButton(
            "处理TIFF图像", "./resource/icons/tiff.png", self
        )
        self.process_tomos_button.setFixedSize(200, 70)
        set_button_role(self.process_tomos_button, "primary")
        self.process_tomos_button.clicked.connect(self.process_tomos)

        self.process_blackline_only_button = IconTextButton(
            "仅去黑线", "./resource/icons/change.png", self
        )
        self.process_blackline_only_button.setFixedSize(200, 70)
        set_button_role(self.process_blackline_only_button, "primary")
        self.process_blackline_only_button.clicked.connect(self.process_blackline_only)

        self.process_grad_button = IconTextButton(
            "梯度方法处理", "./resource/icons/tidu.png", self
        )
        self.process_grad_button.setFixedSize(200, 70)
        set_button_role(self.process_grad_button, "primary")
        self.process_grad_button.clicked.connect(self.process_img_inGrad)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(8)
        action_buttons = [
            self.select_input_button,
            self.select_output_button,
            self.cancel_button,
            self.process_background_button,
            self.process_tomos_button,
            self.process_blackline_only_button,
            self.process_grad_button,
        ]
        for idx, btn in enumerate(action_buttons):
            row = idx // 2
            col = idx % 2
            action_grid.addWidget(btn, row, col)
        controls_layout.addLayout(action_grid)

        gpu_settings_row = QHBoxLayout()
        gpu_settings_row.setSpacing(16)

        gpu_left_column = QVBoxLayout()
        gpu_left_column.setSpacing(8)
        gpu_left_column.addWidget(QLabel("背底处理方法:"))
        gpu_left_column.addWidget(self.process_method_combo)

        gpu_left_column.addWidget(QLabel("GPU 参数:"))
        self.gpu_memory_label = QLabel("显存使用率: 95%")
        self.gpu_memory_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        gpu_left_column.addWidget(self.gpu_memory_label)

        self.stream_count_label = QLabel("CUDA流数量: 2")
        self.stream_count_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        gpu_left_column.addWidget(self.stream_count_label)

        gpu_left_column.addWidget(QLabel("批次大小 (张):"))
        batch_slider_row = QHBoxLayout()
        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(200)
        self.batch_size_slider.setValue(100)
        self.batch_size_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_size_slider.setTickInterval(10)
        self.batch_size_slider.setFixedSize(220, 30)
        self.batch_size_slider.valueChanged.connect(self.on_batch_size_changed)
        batch_slider_row.addWidget(self.batch_size_slider, 1)

        self.batch_size_label = QLabel("100")
        self.batch_size_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        self.batch_size_label.setFixedWidth(30)
        batch_slider_row.addWidget(self.batch_size_label)
        gpu_left_column.addLayout(batch_slider_row)

        self.gpu_settings_button = QPushButton("GPU设置")
        self.gpu_settings_button.setFixedSize(200, 30)
        set_button_role(self.gpu_settings_button, "primary")
        self.gpu_settings_button.clicked.connect(self.open_gpu_settings)
        gpu_left_column.addWidget(self.gpu_settings_button)
        gpu_left_column.addStretch(1)

        self.settings_widget = BlacklineSettingsWidget(self)
        self.settings_widget.setFixedWidth(320)
        gpu_right_column = QVBoxLayout()
        gpu_right_column.addWidget(self.settings_widget, 0, Qt.AlignTop)
        gpu_right_column.addStretch(1)

        gpu_settings_row.addLayout(gpu_left_column, 1)
        gpu_settings_row.addLayout(gpu_right_column, 0)

        controls_layout.addLayout(gpu_settings_row)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("处理日志"))
        right_layout.addWidget(self.text, 2)
        right_layout.addWidget(controls_scroll, 3)

        central_layout.addLayout(right_layout, 1)

        self.setCentralWidget(central_widget)
        self.dock_widget.setWidget(self.tree_view)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget)

        self.current_path = None
        self.input_path = None
        self.output_path = None

    def reset(self):
        pass

    def on_tree_view_clicked(self, index):
        self.current_path = self.model.filePath(index)

    def on_select_input_clicked(self):
        if self.current_path:
            self.text.append(f"输入路径: {self.current_path}\n")
            self.input_path = self.current_path
            self.current_path = None
        else:
            QMessageBox.warning(self, "未选择", "请先在左侧选择文件夹。")

    def on_select_output_clicked(self):
        if self.current_path:
            self.text.append(f"输出路径: {self.current_path}\n")
            self.output_path = self.current_path
            self.current_path = None
        else:
            QMessageBox.warning(self, "未选择", "请先在左侧选择文件夹。")

    def on_cancel_clicked(self):
        self.input_path = None
        self.output_path = None
        QMessageBox.information(self, "已清空", "已清空输入/输出选择。")
        self.text.append("全部路径: 无")
        self.text.append(f"-----------------------------------------------------------")

    def open_gpu_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("GPU设置")
        dialog.setFixedSize(350, 300)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("显存使用率 (%):"))
        memory_spinbox = QSpinBox()
        memory_spinbox.setRange(50, 100)
        memory_spinbox.setValue(int(self.gpu_memory_usage * 100))
        layout.addWidget(memory_spinbox)

        layout.addWidget(QLabel("CUDA流数量:"))
        stream_spinbox = QSpinBox()
        stream_spinbox.setRange(1, 8)
        stream_spinbox.setValue(self.cuda_stream_count)
        layout.addWidget(stream_spinbox)

        layout.addWidget(QLabel("批次大小 (张):"))
        batch_spinbox = QSpinBox()
        batch_spinbox.setRange(1, 500)
        batch_spinbox.setValue(self.batch_size)
        layout.addWidget(batch_spinbox)

        layout.addSpacing(20)

        info_label = QLabel(
            "说明:\n- 显存使用率: 建议使用90-95%\n- CUDA流: 建议使用2-4\n- 批次大小: 建议根据显存大小设置（50-200张）\n- 更高的流数量可以提高并行度\n- 更大的批次可以提高吞吐量但增加显存占用"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info_label)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        cancel_button = QPushButton("取消")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        def apply_settings():
            try:
                memory_value = memory_spinbox.value()
                stream_value = stream_spinbox.value()
                batch_value = batch_spinbox.value()

                if not (50 <= memory_value <= 100):
                    QMessageBox.warning(dialog, "错误", "显存使用率必须在50-100%之间！")
                    return

                if not (1 <= stream_value <= 8):
                    QMessageBox.warning(dialog, "错误", "CUDA流数量必须在1-8之间！")
                    return

                if not (1 <= batch_value <= 500):
                    QMessageBox.warning(dialog, "错误", "批次大小必须在1-500之间！")
                    return

                self.gpu_memory_usage = memory_value / 100.0
                self.cuda_stream_count = stream_value
                self.batch_size = batch_value

                self.gpu_memory_label.setText(f"显存使用率: {memory_value}%")
                self.stream_count_label.setText(f"CUDA流数量: {stream_value}")
                self.batch_size_slider.setValue(batch_value)
                self.batch_size_label.setText(str(batch_value))

                QMessageBox.information(
                    dialog,
                    "成功",
                    f"GPU设置已更新！\n显存使用率: {memory_value}%\nCUDA流数量: {stream_value}\n批次大小: {batch_value}",
                )
                dialog.accept()
            except ValueError:
                QMessageBox.warning(dialog, "错误", "请输入有效的数字！")

        ok_button.clicked.connect(apply_settings)
        cancel_button.clicked.connect(dialog.reject)

        dialog.setLayout(layout)
        dialog.exec_()

    def on_batch_size_changed(self, value):
        self.batch_size = value
        self.batch_size_label.setText(str(value))

    def _is_tiff_file(self, filename):
        lower_name = filename.lower()
        return lower_name.endswith(".tif") or lower_name.endswith(".tiff")

    def _find_keyword_tiff_files(self, folder, keyword, exclude_keywords=None):
        exclude_keywords = [k.lower() for k in (exclude_keywords or [])]
        files = []
        for f in os.listdir(folder):
            lower_name = f.lower()
            if keyword.lower() not in lower_name:
                continue
            if not self._is_tiff_file(f):
                continue
            if any(ex in lower_name for ex in exclude_keywords):
                continue
            files.append(f)
        return sorted(files, key=lambda x: x.lower())

    def process_background(self):
        self.text.append(f"----------------------------------------")
        input_folder = self.input_path
        output_folder = self.output_path
        if not input_folder or not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        input_folder_background = os.path.join(input_folder, "Background")
        if not os.path.exists(input_folder_background):
            QMessageBox.warning(self, "警告", "输入文件夹下缺少 Background 子文件夹！")
            return

        method = self.process_method_combo.currentText()
        self.text.append(f"选择的方法: {method}")

        if method == "单张白减黑":
            self.process_white_black(input_folder_background, output_folder)
        elif method == "多张白减黑均值":
            self.process_average_white_black(input_folder_background, output_folder)
        else:
            QMessageBox.warning(self, "警告", f"未知的方法: {method}")

    def process_white_black(self, input_folder, output_folder):
        self.text.append(f"开始处理单张白减黑...")
        white_files = self._find_keyword_tiff_files(
            input_folder, "white", exclude_keywords=["white_sub_black"]
        )
        black_files = self._find_keyword_tiff_files(
            input_folder, "black", exclude_keywords=["white_sub_black"]
        )

        if not white_files or not black_files:
            QMessageBox.warning(
                self, "警告", "在输入文件夹中未找到合适的白场或黑场文件！"
            )
            return

        if len(white_files) > 1:
            self.text.append(
                f"检测到多个白场文件，按名称排序后使用首个: {white_files[0]}"
            )
        if len(black_files) > 1:
            self.text.append(
                f"检测到多个黑场文件，按名称排序后使用首个: {black_files[0]}"
            )

        self.text.append(f"找到白场文件: {white_files[0]}")
        self.text.append(f"找到黑场文件: {black_files[0]}")

        white_img = GPUImageProcess(os.path.join(input_folder, white_files[0]))
        black_img = GPUImageProcess(os.path.join(input_folder, black_files[0]))

        settings = self.settings_widget.get_settings()

        min_val = settings["min_val"]
        max_val = settings["max_val"]
        grad = settings["grad"]
        blackline_lp = settings["blackline_lp"]

        self.text.append(
            f"参数设置 - 最小值: {min_val}, 最大值: {max_val}, 梯度阈值: {grad}, blackline_lp: {blackline_lp}"
        )

        white_img.min_val = min_val
        white_img.max_val = max_val
        white_img.blackline_lp = blackline_lp
        white_img.valid_grad = grad

        if settings["mode"] == "txt_file":
            columns = settings["columns"]
            if_process_all = False
            self.text.append(f"使用文件读取的坏线: {columns}")
        else:
            if_process_all = True
            columns = None
            self.text.append("使用自动检测模式")

        self.text.append("正在执行白场减黑场...")
        white_img.image_subtract(black_img)

        if columns is not None:
            white_img.If_Process_AllColumns = if_process_all
            white_img.blacklines_columns = columns
            white_img.delete_blacklines_inRange()

        save_dir = os.path.join(output_folder, "Folder1")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "white_sub_black.tif")
        white_img.save_to_path(save_path)

        self.text.append(
            f'<span style="color: green;">成功！处理后的图像已保存到: {save_path}</span>'
        )
        self.text.append(f"-----------------------------------------------------------")

    def process_average_white_black(self, input_folder, output_folder):
        self.text.append(f"开始处理多张白减黑均值...")
        white_files = self._find_keyword_tiff_files(
            input_folder, "white", exclude_keywords=["white_sub_black"]
        )
        black_files = self._find_keyword_tiff_files(
            input_folder, "black", exclude_keywords=["white_sub_black"]
        )

        if not white_files or not black_files:
            QMessageBox.warning(
                self, "警告", "在输入文件夹中未找到合适的白场或黑场文件！"
            )
            return

        self.text.append(f"找到 {len(white_files)} 张白场文件")
        self.text.append(f"找到 {len(black_files)} 张黑场文件")
        self.text.append("白场文件按名称排序后参与均值")
        self.text.append("黑场文件按名称排序后参与均值")

        white_zero_img = GPUImageProcess(os.path.join(input_folder, white_files[0]))
        for white_i in white_files[1:]:
            white_i_img = GPUImageProcess(os.path.join(input_folder, white_i))
            white_zero_img.image_add(white_i_img)

        self_white_img = white_zero_img.img.astype(cp.float32)
        result_white = self_white_img / len(white_files)
        white_zero_img.img = result_white

        save_dir = os.path.join(output_folder, "Folder1")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "AVGwhite.tif")
        white_zero_img.save_to_path(save_path)
        self.text.append(
            f'<span style="color: green;">成功！平均白场已保存到: {save_path}</span>'
        )

        black_zero_img = GPUImageProcess(os.path.join(input_folder, black_files[0]))
        for black_i in black_files[1:]:
            black_i_img = GPUImageProcess(os.path.join(input_folder, black_i))
            black_zero_img.image_add(black_i_img)

        self_black_img = black_zero_img.img.astype(cp.float32)
        result_black = self_black_img / len(black_files)
        black_zero_img.img = result_black

        save_black_path = os.path.join(save_dir, "AVGblack.tif")
        black_zero_img.save_to_path(save_black_path)
        self.text.append(
            f'<span style="color: green;">成功！平均黑场已保存到: {save_black_path}</span>'
        )

        self.process_white_black(save_dir, output_folder)

    def process_tomos(self):
        self.text.append(f"----------------------------------------")
        self.text.append(f"开始处理TIFF图像（GPU批量加速模式）...")
        input_folder = self.input_path
        output_folder = self.output_path
        if not input_folder or not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        input_folder_tomo = os.path.join(input_folder, "Tomos")
        input_folder_background = os.path.join(input_folder, "Background")
        input_folder_wsb = os.path.join(output_folder, "Folder1")

        if not os.path.exists(input_folder_tomo):
            QMessageBox.warning(self, "警告", "输入文件夹下缺少 Tomos 子文件夹！")
            return

        if not os.path.exists(input_folder_background):
            QMessageBox.warning(self, "警告", "输入文件夹下缺少 Background 子文件夹！")
            return

        if not os.path.exists(input_folder_wsb):
            QMessageBox.warning(
                self,
                "警告",
                "输出文件夹下缺少 Folder1（白减黑结果）！请先运行处理背底功能。",
            )
            return

        avg_black_path = os.path.join(input_folder_wsb, "AVGblack.tif")
        if os.path.exists(avg_black_path):
            black_img_path = avg_black_path
            self.text.append("检测到多张均值黑场，优先使用: AVGblack.tif")
        else:
            black_files = self._find_keyword_tiff_files(
                input_folder_background, "black"
            )
            if not black_files:
                QMessageBox.warning(
                    self, "警告", "在 Background 中未找到黑场文件（名称需包含 black）！"
                )
                return
            if len(black_files) > 1:
                self.text.append(
                    f"检测到多个黑场文件，按名称排序后使用首个: {black_files[0]}"
                )
            black_img_path = os.path.join(input_folder_background, black_files[0])
            self.text.append(f"使用黑场文件: {black_files[0]}")

        tomo_files = [
            os.path.join(input_folder_tomo, img)
            for img in os.listdir(input_folder_tomo)
            if self._is_tiff_file(img)
        ]
        tomo_files.sort(key=lambda p: os.path.basename(p).lower())
        self.text.append(f"找到 {len(tomo_files)} 个TIF文件待处理")
        if not tomo_files:
            QMessageBox.warning(
                self, "警告", "Tomos 子文件夹下没有可处理的 .tif/.tiff 文件！"
            )
            return

        preferred_wsb_names = ["white_sub_black.tif", "white_sub_black.tiff"]
        white_sub_black_img_path = None
        for filename in preferred_wsb_names:
            candidate = os.path.join(input_folder_wsb, filename)
            if os.path.exists(candidate):
                white_sub_black_img_path = candidate
                break

        if white_sub_black_img_path is None:
            white_sub_black_files = self._find_keyword_tiff_files(
                input_folder_wsb, "white_sub_black"
            )
            if not white_sub_black_files:
                QMessageBox.warning(self, "警告", "在输出文件夹中未找到白减黑图像！")
                return
            white_sub_black_img_path = os.path.join(
                input_folder_wsb, white_sub_black_files[0]
            )

        self.text.append(
            f"使用白减黑文件: {os.path.basename(white_sub_black_img_path)}"
        )

        settings = self.settings_widget.get_settings()

        if settings["mode"] == "txt_file":
            columns = settings["columns"]
            if_process_all = False
            self.text.append(f"使用文件读取的坏线: {columns}")
            min_val = settings["min_val"]
            max_val = settings["max_val"]
            blackline_lp = settings["blackline_lp"]
        else:
            if_process_all = True
            columns = None
            self.text.append("使用自动检测模式")
            min_val = settings["min_val"]
            max_val = settings["max_val"]
            blackline_lp = settings["blackline_lp"]

        processor = GPUBatchProcessor(
            memory_usage_percent=self.gpu_memory_usage,
            num_streams=self.cuda_stream_count,
            batch_size=self.batch_size,
        )
        processor.text = self.text

        self.start_time = time.time()

        def progress_callback(processed, total, memory_info):
            elapsed = time.time() - self.start_time
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / speed if speed > 0 else 0

            self.text.append(
                f"进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"速度: {speed:.1f}张/秒 | 剩余时间: {remaining:.0f}秒 | "
                f"显存: {memory_info['used_mb']:.1f}MB/{memory_info['total_mb']:.1f}MB "
                f"({memory_info['usage_percent']:.1f}%) | "
                f"批次大小: 自动"
            )

        processor.set_progress_callback(progress_callback)

        self.text.append("开始步骤1: 减去黑场并去黑线（GPU批量处理）...")

        folder2_dir = os.path.join(output_folder, "Folder2")
        processor.process_tomos_batch(
            tomo_paths=tomo_files,
            black_img_path=black_img_path,
            white_sub_black_img_path=white_sub_black_img_path,
            output_folder=folder2_dir,
            min_val=min_val,
            max_val=max_val,
            blackline_lp=blackline_lp,
            if_process_all=if_process_all,
            blacklines_columns=columns,
        )

        self.text.append(f"步骤1完成！结果保存在: {folder2_dir}")
        self.text.append(f"----------------------------------------")

        self.start_time = time.time()

        def progress_callback_step2(processed, total, memory_info):
            elapsed = time.time() - self.start_time
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / speed if speed > 0 else 0

            self.text.append(
                f"进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"速度: {speed:.1f}张/秒 | 剩余时间: {remaining:.0f}秒 | "
                f"显存: {memory_info['used_mb']:.1f}MB/{memory_info['total_mb']:.1f}MB "
                f"({memory_info['usage_percent']:.1f}%)"
            )

        processor.set_progress_callback(progress_callback_step2)

        folder3_dir = os.path.join(output_folder, "Folder3")
        tomo_result_files = [
            os.path.join(folder2_dir, f)
            for f in os.listdir(folder2_dir)
            if f.endswith((".tiff", ".tif"))
        ]

        self.text.append(f"找到 {len(tomo_result_files)} 个处理后的文件")
        self.text.append("开始步骤2: 除以白减黑（GPU批量处理）...")

        processor.process_divide_batch(
            tomo_result_files,
            white_sub_black_img_path,
            folder3_dir,
            progress_callback_step2,
        )

        self.text.append(
            f'<span style="color: green;">处理完成！结果保存在: {folder3_dir}</span>'
        )
        self.text.append(f"-----------------------------------------------------------")

    def process_blackline_only(self):
        self.text.append(f"----------------------------------------")
        self.text.append("开始仅去黑线处理（不做背底校正）...")

        input_folder = self.input_path
        output_folder = self.output_path
        if not input_folder or not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹。")
            return

        image_todo_files = [
            os.path.join(input_folder, img)
            for img in os.listdir(input_folder)
            if self._is_tiff_file(img)
        ]
        image_todo_files.sort(key=lambda p: os.path.basename(p).lower())

        if not image_todo_files:
            QMessageBox.warning(self, "警告", "输入目录中未找到 .tif/.tiff 文件。")
            return

        settings = self.settings_widget.get_settings()
        min_val = settings["min_val"]
        max_val = settings["max_val"]
        blackline_lp = settings["blackline_lp"]

        if settings["mode"] == "txt_file":
            columns = settings.get("columns") or []
            if_process_all = False
            if not columns:
                QMessageBox.warning(
                    self,
                    "警告",
                    "未读取到坏线列数据，请选择 txt 文件或切换到自动检测模式。",
                )
                return
            self.text.append(f"模式: txt_file，坏线列数量: {len(columns)}")
        else:
            if_process_all = True
            columns = None
            self.text.append("模式: auto_detect（全列检测）")

        self.text.append(f"待处理图像数量: {len(image_todo_files)}")
        self.text.append(
            f"参数 - 最小值: {min_val}, 最大值: {max_val}, blackline_lp: {blackline_lp}"
        )

        save_dir = os.path.join(output_folder, "Folder_blackline_only")
        os.makedirs(save_dir, exist_ok=True)
        processor = GPUBatchProcessor(
            memory_usage_percent=self.gpu_memory_usage,
            num_streams=self.cuda_stream_count,
            batch_size=self.batch_size,
        )
        processor.text = self.text

        self.start_time = time.time()

        def progress_callback(processed, total, memory_info):
            elapsed = time.time() - self.start_time
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / speed if speed > 0 else 0
            self.text.append(
                f"进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"速度: {speed:.1f}张/秒 | 剩余时间: {remaining:.0f}秒 | "
                f"显存: {memory_info['used_mb']:.1f}MB/{memory_info['total_mb']:.1f}MB "
                f"({memory_info['usage_percent']:.1f}%)"
            )

        processor.set_progress_callback(progress_callback)
        success_count, fail_count = processor.process_blackline_only_batch(
            input_paths=image_todo_files,
            output_folder=save_dir,
            min_val=min_val,
            max_val=max_val,
            blackline_lp=blackline_lp,
            if_process_all=if_process_all,
            blacklines_columns=columns,
            progress_callback=progress_callback,
        )

        self.text.append(
            f'<span style="color: green;">仅去黑线处理完成。成功: {success_count}, '
            f"失败: {fail_count}, 输出目录: {save_dir}</span>"
        )
        self.text.append(f"-----------------------------------------------------------")

    def process_img_inGrad(self):
        self.text.append(f"----------------------------------------")
        self.text.append(f"开始梯度方法处理...")
        input_folder = self.input_path
        output_folder = self.output_path
        if not input_folder or not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输入和输出文件夹！")
            return

        image_todo_files = [
            os.path.join(input_folder, img)
            for img in os.listdir(input_folder)
            if img.endswith(".tif") or img.endswith(".tiff")
        ]
        self.text.append(f"找到 {len(image_todo_files)} 个图像文件待处理")

        settings = self.settings_widget.get_settings()

        if settings["mode"] == "txt_file":
            columns = settings["columns"]
            if_process_all = False
            self.text.append(f"使用文件读取的坏线: {columns}")
            grad = settings["grad"]
            blackline_lp = settings["blackline_lp"]
        else:
            if_process_all = True
            columns = None
            self.text.append("使用自动检测模式")
            grad = settings["grad"]
            blackline_lp = settings["blackline_lp"]

        for image_todo in tqdm.tqdm(image_todo_files, desc="梯度处理"):
            try:
                image_todo_img = GPUImageProcess(image_todo)
                image_todo_img.valid_grad = grad
                image_todo_img.blackline_lp = blackline_lp
                image_todo_img.If_Process_AllColumns = if_process_all
                image_todo_img.blacklines_columns = columns
                image_todo_img.delete_blacklines_inGrad()

                save_dir = os.path.join(output_folder, "Folder_inGrad")
                os.makedirs(save_dir, exist_ok=True)
                basename = os.path.basename(image_todo)
                if basename.endswith(".tiff"):
                    save_path = os.path.join(
                        save_dir, basename.replace(".tiff", "_.tiff")
                    )
                elif basename.endswith(".tif"):
                    save_path = os.path.join(
                        save_dir, basename.replace(".tif", "_.tif")
                    )
                else:
                    save_path = os.path.join(save_dir, basename)
                image_todo_img.save_to_path(save_path)
                self.text.append(
                    f'<span style="color: green;">成功: {os.path.basename(image_todo)}</span>'
                )

            except IOError:
                self.text.append(
                    f'<span style="color: red;">  警告: 无法处理文件 {image_todo}，跳过...</span>'
                )

        self.text.append(
            f'<span style="color: green;">梯度方法处理完成！结果保存在: {os.path.join(output_folder, "Folder_inGrad")}</span>'
        )
        self.text.append(f"-----------------------------------------------------------")
