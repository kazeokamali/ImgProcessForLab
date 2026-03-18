import os
from typing import Optional, Tuple

import imageio.v3 as iio
import numpy as np
import tifffile as tiff
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.lifton2019.bad_pixel_mask import build_bad_pixel_mask
from src.lifton2019.bad_pixel_repair import repair_bad_pixels
from src.lifton2019.io_loader import (
    collect_image_files,
    load_bad_pixel_calibration_set,
    load_image,
)
from src.lifton2019.models import BadPixelConfig, ProcessingConfig
from src.interfaces.ui_theme import apply_interface_theme, set_button_role


class BadPixelInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("badPixelInterface")
        self.setWindowTitle("坏点掩膜与修复")
        self.resize(1600, 900)
        apply_interface_theme(self)

        self.current_mask: Optional[np.ndarray] = None
        self.current_mask_path: str = ""

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(220)

        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        root_layout.addWidget(log_group, 2)

        controls_widget = QWidget()
        controls_layout = QGridLayout(controls_widget)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(12)

        folder_group = self._build_folder_group()
        calib_group = self._build_calib_group()
        detection_group = self._build_detection_group()
        repair_group = self._build_repair_group()
        action_group = self._build_action_group()

        note_label = QLabel(
            "独立坏点处理流程：\n"
            "1) 基于单组 dark/flat 数据生成统一坏点掩膜\n"
            "2) 使用该掩膜修复投影图像"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")

        controls_layout.addWidget(folder_group, 0, 0, 1, 2)
        controls_layout.addWidget(calib_group, 1, 0)
        controls_layout.addWidget(detection_group, 1, 1)
        controls_layout.addWidget(repair_group, 2, 0)
        controls_layout.addWidget(action_group, 2, 1)
        controls_layout.addWidget(note_label, 3, 0, 1, 2)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setRowStretch(4, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        root_layout.addWidget(controls_scroll, 3)

        self.setCentralWidget(central_widget)

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox("目录设置")
        layout = QVBoxLayout()

        self.dark_edit = self._build_folder_row(
            layout, "Dark 目录", self.on_select_dark_folder
        )
        self.flat_edit = self._build_folder_row(
            layout, "Flat 目录", self.on_select_flat_folder
        )
        self.projection_edit = self._build_folder_row(
            layout, "投影目录", self.on_select_projection_folder
        )
        self.output_edit = self._build_folder_row(
            layout, "输出目录", self.on_select_output_folder
        )

        self.known_mask_edit = self._build_file_row(
            layout, "已知掩膜文件（可选）", self.on_select_known_mask_file
        )
        self.known_badline_edit = self._build_file_row(
            layout, "已知坏线文件（可选）", self.on_select_known_badline_file
        )

        group.setLayout(layout)
        return group

    def _build_calib_group(self) -> QGroupBox:
        group = QGroupBox("标定参数")
        layout = QFormLayout()

        self.raw_width_spin = QSpinBox()
        self.raw_width_spin.setRange(1, 100000)
        self.raw_width_spin.setValue(2340)
        layout.addRow("RAW 宽度:", self.raw_width_spin)

        self.raw_height_spin = QSpinBox()
        self.raw_height_spin.setRange(1, 100000)
        self.raw_height_spin.setValue(2882)
        layout.addRow("RAW 高度:", self.raw_height_spin)

        group.setLayout(layout)
        return group

    def _build_detection_group(self) -> QGroupBox:
        group = QGroupBox("检测参数")
        layout = QFormLayout()

        self.enable_flat_neighbor_cb = QCheckBox("启用 Flat 邻域偏差检测")
        self.enable_flat_neighbor_cb.setChecked(True)
        layout.addRow(self.enable_flat_neighbor_cb)

        self.enable_dark_neighbor_cb = QCheckBox("启用 Dark 邻域偏差检测")
        self.enable_dark_neighbor_cb.setChecked(True)
        layout.addRow(self.enable_dark_neighbor_cb)

        self.enable_stability_cb = QCheckBox("启用稳定性检测")
        self.enable_stability_cb.setChecked(True)
        layout.addRow(self.enable_stability_cb)

        self.neighborhood_size_spin = QSpinBox()
        self.neighborhood_size_spin.setRange(3, 21)
        self.neighborhood_size_spin.setSingleStep(2)
        self.neighborhood_size_spin.setValue(3)
        layout.addRow("邻域窗口大小:", self.neighborhood_size_spin)

        self.flat_sigma_spin = QDoubleSpinBox()
        self.flat_sigma_spin.setRange(1.0, 50.0)
        self.flat_sigma_spin.setValue(8.0)
        layout.addRow("Flat 检测阈值(sigma):", self.flat_sigma_spin)

        self.dark_sigma_spin = QDoubleSpinBox()
        self.dark_sigma_spin.setRange(1.0, 50.0)
        self.dark_sigma_spin.setValue(8.0)
        layout.addRow("Dark 检测阈值(sigma):", self.dark_sigma_spin)

        self.stability_sigma_spin = QDoubleSpinBox()
        self.stability_sigma_spin.setRange(1.0, 50.0)
        self.stability_sigma_spin.setValue(6.0)
        layout.addRow("稳定性阈值(sigma):", self.stability_sigma_spin)

        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(0, 10)
        self.dilation_spin.setValue(0)
        layout.addRow("掩膜膨胀半径:", self.dilation_spin)

        self.min_component_spin = QSpinBox()
        self.min_component_spin.setRange(1, 1000000)
        self.min_component_spin.setValue(1)
        layout.addRow("最小连通域大小:", self.min_component_spin)

        group.setLayout(layout)
        return group

    def _build_repair_group(self) -> QGroupBox:
        group = QGroupBox("修复参数")
        layout = QFormLayout()

        self.repair_window_spin = QSpinBox()
        self.repair_window_spin.setRange(3, 21)
        self.repair_window_spin.setSingleStep(2)
        self.repair_window_spin.setValue(3)
        layout.addRow("修复窗口大小:", self.repair_window_spin)

        self.repair_iter_spin = QSpinBox()
        self.repair_iter_spin.setRange(1, 20)
        self.repair_iter_spin.setValue(6)
        layout.addRow("修复迭代次数:", self.repair_iter_spin)

        self.enable_directional_cb = QCheckBox("启用定向坏线修复")
        self.enable_directional_cb.setChecked(True)
        layout.addRow(self.enable_directional_cb)

        self.directional_ratio_spin = QDoubleSpinBox()
        self.directional_ratio_spin.setRange(1.0, 20.0)
        self.directional_ratio_spin.setValue(6.0)
        layout.addRow("线状判定长宽比:", self.directional_ratio_spin)

        group.setLayout(layout)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("执行操作")
        layout = QVBoxLayout()

        self.validate_btn = QPushButton("校验 dark/flat 目录")
        set_button_role(self.validate_btn, "primary")
        self.validate_btn.clicked.connect(self.on_validate_clicked)
        layout.addWidget(self.validate_btn)

        self.build_mask_btn = QPushButton("生成坏点掩膜")
        set_button_role(self.build_mask_btn, "primary")
        self.build_mask_btn.clicked.connect(self.on_build_mask_clicked)
        layout.addWidget(self.build_mask_btn)

        self.repair_btn = QPushButton("修复投影目录")
        set_button_role(self.repair_btn, "primary")
        self.repair_btn.clicked.connect(self.on_repair_clicked)
        layout.addWidget(self.repair_btn)

        clear_btn = QPushButton("清空日志")
        set_button_role(clear_btn, "danger")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        group.setLayout(layout)
        return group

    def _build_folder_row(self, parent_layout: QVBoxLayout, label_text: str, callback) -> QLineEdit:
        row = QHBoxLayout()
        label = QLabel(label_text)
        edit = QLineEdit()
        edit.setReadOnly(True)
        button = QPushButton("选择")
        button.clicked.connect(callback)
        row.addWidget(label)
        row.addWidget(edit)
        row.addWidget(button)
        parent_layout.addLayout(row)
        return edit

    def _build_file_row(self, parent_layout: QVBoxLayout, label_text: str, callback) -> QLineEdit:
        row = QHBoxLayout()
        label = QLabel(label_text)
        edit = QLineEdit()
        edit.setReadOnly(True)
        button = QPushButton("选择")
        button.clicked.connect(callback)
        row.addWidget(label)
        row.addWidget(edit)
        row.addWidget(button)
        parent_layout.addLayout(row)
        return edit

    def _select_folder(self, title: str, target_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, title, "")
        if folder:
            target_edit.setText(folder)
            self._log(f"{title}: {folder}")

    def _select_file(self, title: str, target_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, title, "", "所有文件 (*.*)")
        if path:
            target_edit.setText(path)
            self._log(f"{title}: {path}")

    def on_select_dark_folder(self):
        self._select_folder("选择 Dark 目录", self.dark_edit)

    def on_select_flat_folder(self):
        self._select_folder("选择 Flat 目录", self.flat_edit)

    def on_select_projection_folder(self):
        self._select_folder("选择投影目录", self.projection_edit)

    def on_select_output_folder(self):
        self._select_folder("选择输出目录", self.output_edit)

    def on_select_known_mask_file(self):
        self._select_file("选择已知掩膜文件", self.known_mask_edit)

    def on_select_known_badline_file(self):
        self._select_file("选择已知坏线文件", self.known_badline_edit)

    def _build_processing_config(self) -> ProcessingConfig:
        bad_cfg = BadPixelConfig(
            enabled=True,
            enable_flat_neighbor_check=self.enable_flat_neighbor_cb.isChecked(),
            enable_dark_neighbor_check=self.enable_dark_neighbor_cb.isChecked(),
            enable_stability_check=self.enable_stability_cb.isChecked(),
            known_mask_path=self.known_mask_edit.text().strip(),
            known_badline_path=self.known_badline_edit.text().strip(),
            neighborhood_size=int(self.neighborhood_size_spin.value()),
            flat_neighbor_sigma=float(self.flat_sigma_spin.value()),
            dark_neighbor_sigma=float(self.dark_sigma_spin.value()),
            stability_sigma=float(self.stability_sigma_spin.value()),
            dilation_radius=int(self.dilation_spin.value()),
            min_component_size=int(self.min_component_spin.value()),
            repair_window_size=int(self.repair_window_spin.value()),
            repair_iterations=int(self.repair_iter_spin.value()),
            enable_directional_line_repair=self.enable_directional_cb.isChecked(),
            directional_line_aspect_ratio=float(self.directional_ratio_spin.value()),
        )
        return ProcessingConfig(
            num_points=1,
            raw_width=int(self.raw_width_spin.value()),
            raw_height=int(self.raw_height_spin.value()),
            point_pattern=r".*",
            bad_pixel=bad_cfg,
        )

    def _required_calib_paths(self, require_output: bool = True) -> Optional[Tuple[str, str, str]]:
        dark_folder = self.dark_edit.text().strip()
        flat_folder = self.flat_edit.text().strip()
        output_folder = self.output_edit.text().strip()

        if not dark_folder or not flat_folder:
            QMessageBox.warning(self, "警告", "请先选择 dark/flat 目录。")
            return None

        if require_output and not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出目录。")
            return None

        return dark_folder, flat_folder, output_folder

    def on_validate_clicked(self):
        maybe = self._required_calib_paths(require_output=False)
        if maybe is None:
            return

        dark_folder, flat_folder, _ = maybe
        cfg = self._build_processing_config()
        compute_std_maps = cfg.bad_pixel.enable_stability_check

        try:
            calib = load_bad_pixel_calibration_set(
                dark_folder=dark_folder,
                flat_folder=flat_folder,
                raw_shape=(cfg.raw_height, cfg.raw_width),
                compute_std_maps=compute_std_maps,
            )
            self._log("校验通过。")
            self._log(
                f"图像尺寸: {calib.dark_avg.shape} | dark帧数: {calib.frame_counts.get('dark', 0)} | "
                f"flat帧数: {calib.frame_counts.get('flat', 0)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "校验失败", str(e))
            self._log(f"校验失败: {e}")

    def on_build_mask_clicked(self):
        maybe = self._required_calib_paths()
        if maybe is None:
            return

        dark_folder, flat_folder, output_folder = maybe
        cfg = self._build_processing_config()
        compute_std_maps = cfg.bad_pixel.enable_stability_check

        try:
            calib = load_bad_pixel_calibration_set(
                dark_folder=dark_folder,
                flat_folder=flat_folder,
                raw_shape=(cfg.raw_height, cfg.raw_width),
                compute_std_maps=compute_std_maps,
            )

            mask, stats = build_bad_pixel_mask(calib, cfg.bad_pixel)
            self.current_mask = mask

            os.makedirs(output_folder, exist_ok=True)
            tif_path = os.path.join(output_folder, "bad_pixel_mask.tif")
            npy_path = os.path.join(output_folder, "bad_pixel_mask.npy")
            tiff.imwrite(tif_path, (mask.astype(np.uint8) * 255), dtype=np.uint8)
            np.save(npy_path, mask.astype(np.bool_))
            self.current_mask_path = tif_path

            self._log(f"掩膜生成完成，总坏点数: {stats.get('total', 0)}")
            self._log(f"掩膜已保存: {tif_path}")
            self._log(f"掩膜已保存: {npy_path}")
            QMessageBox.information(
                self,
                "完成",
                f"坏点掩膜生成完成。\n总坏点数: {stats.get('total', 0)}",
            )
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            self._log(f"生成掩膜失败: {e}")

    def _load_mask_for_repair(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.current_mask is not None and self.current_mask.shape == shape:
            return self.current_mask

        known = self.known_mask_edit.text().strip()
        if not known or not os.path.exists(known):
            raise ValueError("没有可用坏点掩膜。请先生成掩膜，或选择已知掩膜文件。")

        if known.lower().endswith(".npy"):
            arr = np.load(known)
        else:
            arr = iio.imread(known)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
        arr = np.asarray(arr)
        if arr.shape != shape:
            raise ValueError(f"掩膜尺寸不匹配：期望 {shape}，实际 {arr.shape}")
        return (arr > 0)

    def on_repair_clicked(self):
        projection_folder = self.projection_edit.text().strip()
        output_folder = self.output_edit.text().strip()
        if not projection_folder or not output_folder:
            QMessageBox.warning(self, "警告", "请先选择投影目录和输出目录。")
            return

        cfg = self._build_processing_config()
        proj_paths = collect_image_files(projection_folder)
        if not proj_paths:
            QMessageBox.warning(self, "警告", "投影目录中未找到可处理图像文件。")
            return

        out_dir = os.path.join(output_folder, "projections_badpixel_repaired")
        os.makedirs(out_dir, exist_ok=True)

        total = len(proj_paths)
        success = 0
        fail = 0
        self.progress_bar.setValue(0)

        try:
            first = load_image(proj_paths[0], (cfg.raw_height, cfg.raw_width))
            mask = self._load_mask_for_repair(first.shape)

            for idx, path in enumerate(proj_paths, start=1):
                name = os.path.basename(path)
                try:
                    img = load_image(path, (cfg.raw_height, cfg.raw_width))
                    repaired = repair_bad_pixels(
                        img,
                        mask,
                        window_size=cfg.bad_pixel.repair_window_size,
                        max_iterations=cfg.bad_pixel.repair_iterations,
                        enable_directional_line_repair=cfg.bad_pixel.enable_directional_line_repair,
                        directional_line_aspect_ratio=cfg.bad_pixel.directional_line_aspect_ratio,
                    )
                    stem, _ = os.path.splitext(name)
                    save_path = os.path.join(out_dir, f"{stem}.tif")
                    tiff.imwrite(save_path, repaired.astype(np.float32), dtype=np.float32)
                    success += 1
                except Exception as e:
                    fail += 1
                    self._log(f"修复失败: {name}, 原因: {e}")

                percent = int((idx / total) * 100)
                self.progress_bar.setValue(percent)
                if idx % 10 == 0 or idx == total:
                    self._log(f"修复进度: {idx}/{total}")
                QApplication.processEvents()

            self._log(f"修复完成。成功={success}, 失败={fail}, 输出目录={out_dir}")
            QMessageBox.information(
                self,
                "完成",
                f"修复完成。\n成功: {success}\n失败: {fail}",
            )
        except Exception as e:
            QMessageBox.critical(self, "修复失败", str(e))
            self._log(f"修复失败: {e}")

    def _log(self, message: str):
        self.log_text.append(message)
