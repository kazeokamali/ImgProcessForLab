import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.utils.ring_artifact_processing import (
    PolarTransformer,
    remove_ring_artifact_frequency,
    remove_ring_artifact_morphology,
    remove_ring_artifact_polar,
)
from src.interfaces.ui_theme import apply_interface_theme, set_button_role

RECOMMENDED_PRESET_V1 = {
    "raw_width": 2340,
    "raw_height": 2882,
    "center_mode": 0,
    "center_x": 1170.0,
    "center_y": 1441.0,
    "method": 0,
    "num_angles": 2048,
    "polar_sigma": 7.0,
    "polar_strength": 0.9,
    "freq_cutoff": 4,
    "freq_suppression": 0.7,
    "freq_notch": 0,
    "freq_notch_width": 1,
    "morph_opening": 121,
    "morph_sigma": 6.0,
    "morph_strength": 0.8,
}


class RingArtifactInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ringArtifactInterface")
        self.setWindowTitle("环形伪影后处理")
        self.resize(1500, 900)
        apply_interface_theme(self)

        self.input_folder: Optional[str] = None
        self.output_folder: Optional[str] = None
        self.transformer = PolarTransformer()

        self.custom_templates: Dict[str, dict] = {}
        self.template_store_path = self._resolve_template_store_path()

        self._init_ui()
        self._load_custom_templates()
        self._refresh_template_combo()

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        left_panel.addWidget(self._build_folder_group())
        left_panel.addWidget(self._build_raw_group())
        left_panel.addWidget(self._build_center_group())
        left_panel.addWidget(self._build_method_group())
        left_panel.addWidget(self._build_template_group())
        left_panel.addWidget(self._build_action_group())

        note_label = QLabel(
            "说明:\n"
            "1. 该界面用于对 3D 重构切片做环形伪影后处理。\n"
            "2. 支持 tif/tiff/raw 输入，输出统一为 float32 tif。\n"
            "3. 推荐先使用“极坐标变换法”，再尝试频域或形态学参数。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")
        left_panel.addWidget(note_label)
        left_panel.addStretch(1)

        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        self.setCentralWidget(central_widget)

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox("输入输出目录")
        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.input_line_edit = QLineEdit()
        self.input_line_edit.setReadOnly(True)
        input_btn = QPushButton("选择输入文件夹")
        input_btn.clicked.connect(self.on_select_input_folder)
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(input_btn)
        layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.output_line_edit = QLineEdit()
        self.output_line_edit.setReadOnly(True)
        output_btn = QPushButton("选择输出文件夹")
        output_btn.clicked.connect(self.on_select_output_folder)
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)

        group.setLayout(layout)
        return group

    def _build_raw_group(self) -> QGroupBox:
        group = QGroupBox("RAW 尺寸设置")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("宽:"))
        self.raw_width_spin = QSpinBox()
        self.raw_width_spin.setRange(1, 100000)
        self.raw_width_spin.setValue(2340)
        self.raw_width_spin.valueChanged.connect(self._on_raw_size_changed)
        layout.addWidget(self.raw_width_spin)

        layout.addWidget(QLabel("高:"))
        self.raw_height_spin = QSpinBox()
        self.raw_height_spin.setRange(1, 100000)
        self.raw_height_spin.setValue(2882)
        self.raw_height_spin.valueChanged.connect(self._on_raw_size_changed)
        layout.addWidget(self.raw_height_spin)

        layout.addStretch(1)
        group.setLayout(layout)
        return group

    def _build_center_group(self) -> QGroupBox:
        group = QGroupBox("重建中心")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("模式:"))
        self.center_mode_combo = QComboBox()
        self.center_mode_combo.addItems(["自动中心", "手动中心"])
        self.center_mode_combo.currentIndexChanged.connect(self._on_center_mode_changed)
        layout.addWidget(self.center_mode_combo)

        layout.addWidget(QLabel("中心 X:"))
        self.center_x_spin = QDoubleSpinBox()
        self.center_x_spin.setRange(0.0, 100000.0)
        self.center_x_spin.setDecimals(2)
        self.center_x_spin.setValue(self.raw_width_spin.value() / 2.0)
        self.center_x_spin.setEnabled(False)
        layout.addWidget(self.center_x_spin)

        layout.addWidget(QLabel("中心 Y:"))
        self.center_y_spin = QDoubleSpinBox()
        self.center_y_spin.setRange(0.0, 100000.0)
        self.center_y_spin.setDecimals(2)
        self.center_y_spin.setValue(self.raw_height_spin.value() / 2.0)
        self.center_y_spin.setEnabled(False)
        layout.addWidget(self.center_y_spin)

        layout.addStretch(1)
        group.setLayout(layout)
        return group

    def _build_method_group(self) -> QGroupBox:
        group = QGroupBox("去环方法与参数")
        layout = QVBoxLayout()

        common_layout = QHBoxLayout()
        common_layout.addWidget(QLabel("方法:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["极坐标变换法", "频域滤波法", "形态学去环法"])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        common_layout.addWidget(self.method_combo)

        common_layout.addWidget(QLabel("极角采样数:"))
        self.num_angles_spin = QSpinBox()
        self.num_angles_spin.setRange(360, 8192)
        self.num_angles_spin.setSingleStep(64)
        self.num_angles_spin.setValue(2048)
        common_layout.addWidget(self.num_angles_spin)

        common_layout.addStretch(1)
        layout.addLayout(common_layout)

        self.method_stacked = QStackedWidget()
        self.method_stacked.addWidget(self._build_polar_page())
        self.method_stacked.addWidget(self._build_frequency_page())
        self.method_stacked.addWidget(self._build_morphology_page())
        layout.addWidget(self.method_stacked)

        group.setLayout(layout)
        return group

    def _build_polar_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        layout.addWidget(QLabel("条纹平滑 sigma:"))
        self.polar_sigma_spin = QDoubleSpinBox()
        self.polar_sigma_spin.setRange(0.1, 200.0)
        self.polar_sigma_spin.setDecimals(2)
        self.polar_sigma_spin.setValue(6.0)
        layout.addWidget(self.polar_sigma_spin)

        layout.addWidget(QLabel("校正强度:"))
        self.polar_strength_spin = QDoubleSpinBox()
        self.polar_strength_spin.setRange(0.0, 3.0)
        self.polar_strength_spin.setDecimals(2)
        self.polar_strength_spin.setValue(1.0)
        self.polar_strength_spin.setSingleStep(0.1)
        layout.addWidget(self.polar_strength_spin)

        layout.addStretch(1)
        return page

    def _build_frequency_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        layout.addWidget(QLabel("低频截止:"))
        self.freq_cutoff_spin = QSpinBox()
        self.freq_cutoff_spin.setRange(0, 128)
        self.freq_cutoff_spin.setValue(3)
        layout.addWidget(self.freq_cutoff_spin)

        layout.addWidget(QLabel("抑制比例:"))
        self.freq_suppression_spin = QDoubleSpinBox()
        self.freq_suppression_spin.setRange(0.0, 1.0)
        self.freq_suppression_spin.setDecimals(2)
        self.freq_suppression_spin.setValue(0.7)
        self.freq_suppression_spin.setSingleStep(0.05)
        layout.addWidget(self.freq_suppression_spin)

        layout.addWidget(QLabel("周期频点(0关闭):"))
        self.freq_notch_spin = QSpinBox()
        self.freq_notch_spin.setRange(0, 512)
        self.freq_notch_spin.setValue(0)
        layout.addWidget(self.freq_notch_spin)

        layout.addWidget(QLabel("陷波宽度:"))
        self.freq_notch_width_spin = QSpinBox()
        self.freq_notch_width_spin.setRange(1, 32)
        self.freq_notch_width_spin.setValue(1)
        layout.addWidget(self.freq_notch_width_spin)

        layout.addStretch(1)
        return page

    def _build_morphology_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        layout.addWidget(QLabel("开运算窗口(角向):"))
        self.morph_opening_spin = QSpinBox()
        self.morph_opening_spin.setRange(3, 2001)
        self.morph_opening_spin.setSingleStep(2)
        self.morph_opening_spin.setValue(101)
        layout.addWidget(self.morph_opening_spin)

        layout.addWidget(QLabel("轮廓平滑 sigma:"))
        self.morph_sigma_spin = QDoubleSpinBox()
        self.morph_sigma_spin.setRange(0.1, 200.0)
        self.morph_sigma_spin.setDecimals(2)
        self.morph_sigma_spin.setValue(6.0)
        layout.addWidget(self.morph_sigma_spin)

        layout.addWidget(QLabel("校正强度:"))
        self.morph_strength_spin = QDoubleSpinBox()
        self.morph_strength_spin.setRange(0.0, 3.0)
        self.morph_strength_spin.setDecimals(2)
        self.morph_strength_spin.setValue(1.0)
        self.morph_strength_spin.setSingleStep(0.1)
        layout.addWidget(self.morph_strength_spin)

        layout.addStretch(1)
        return page

    def _build_template_group(self) -> QGroupBox:
        group = QGroupBox("预设与模板")
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        preset_btn = QPushButton("一键预设 V1")
        preset_btn.clicked.connect(self.on_apply_recommended_preset)
        top_layout.addWidget(preset_btn)
        top_layout.addStretch(1)
        layout.addLayout(top_layout)

        apply_layout = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(280)
        apply_layout.addWidget(self.template_combo)

        apply_btn = QPushButton("应用模板")
        apply_btn.clicked.connect(self.on_apply_selected_template)
        apply_layout.addWidget(apply_btn)

        delete_btn = QPushButton("删除模板")
        delete_btn.clicked.connect(self.on_delete_selected_template)
        apply_layout.addWidget(delete_btn)
        layout.addLayout(apply_layout)

        save_layout = QHBoxLayout()
        self.template_name_edit = QLineEdit()
        self.template_name_edit.setPlaceholderText("输入模板名后保存")
        save_layout.addWidget(self.template_name_edit)

        save_btn = QPushButton("保存为模板")
        save_btn.clicked.connect(self.on_save_template)
        save_layout.addWidget(save_btn)
        layout.addLayout(save_layout)

        group.setLayout(layout)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("执行")
        layout = QHBoxLayout()

        run_btn = QPushButton("开始批量去环")
        set_button_role(run_btn, "primary")
        run_btn.clicked.connect(self.on_run_batch)
        layout.addWidget(run_btn)

        clear_btn = QPushButton("清空日志")
        set_button_role(clear_btn, "danger")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    def _on_raw_size_changed(self):
        if self.center_mode_combo.currentIndex() == 0:
            self.center_x_spin.setValue(self.raw_width_spin.value() / 2.0)
            self.center_y_spin.setValue(self.raw_height_spin.value() / 2.0)

    def _on_center_mode_changed(self, index: int):
        manual = index == 1
        self.center_x_spin.setEnabled(manual)
        self.center_y_spin.setEnabled(manual)

    def _on_method_changed(self, index: int):
        self.method_stacked.setCurrentIndex(index)

    def on_select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入切片文件夹", "")
        if folder:
            self.input_folder = folder
            self.input_line_edit.setText(folder)
            self._log(f"输入目录: {folder}")

    def on_select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", "")
        if folder:
            self.output_folder = folder
            self.output_line_edit.setText(folder)
            self._log(f"输出目录: {folder}")

    def on_apply_recommended_preset(self):
        self._apply_template_payload(RECOMMENDED_PRESET_V1)
        self._log("已应用推荐预设 V1")

    def on_save_template(self):
        name = self.template_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请先输入模板名称。")
            return

        payload = self._collect_template_payload()
        if name in self.custom_templates:
            choice = QMessageBox.question(
                self,
                "确认覆盖",
                f"模板“{name}”已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        self.custom_templates[name] = payload
        if not self._save_custom_templates():
            QMessageBox.critical(self, "错误", "模板保存失败，请检查日志。")
            return

        self._refresh_template_combo(selected_name=name)
        self._log(f"模板已保存: {name}")

    def on_apply_selected_template(self):
        if self.template_combo.count() == 0:
            QMessageBox.warning(self, "提示", "暂无可用模板。")
            return

        name = self.template_combo.currentText().strip()
        payload = self.custom_templates.get(name)
        if payload is None:
            QMessageBox.warning(self, "提示", "模板不存在或已损坏。")
            return

        self._apply_template_payload(payload)
        self._log(f"已应用模板: {name}")

    def on_delete_selected_template(self):
        if self.template_combo.count() == 0:
            QMessageBox.warning(self, "提示", "暂无可删除模板。")
            return

        name = self.template_combo.currentText().strip()
        choice = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除模板“{name}”？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return

        self.custom_templates.pop(name, None)
        if not self._save_custom_templates():
            QMessageBox.critical(self, "错误", "模板删除后保存失败，请检查日志。")
            return

        self._refresh_template_combo()
        self._log(f"模板已删除: {name}")

    def on_run_batch(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "提示", "请先选择输入和输出目录。")
            return

        image_paths = self._collect_image_paths(self.input_folder)
        if not image_paths:
            QMessageBox.warning(self, "提示", "输入目录中没有找到 tif/tiff/raw 文件。")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        center_xy = self._get_center_xy()
        method_name = self.method_combo.currentText()
        num_angles = self.num_angles_spin.value()

        self._log("-" * 60)
        self._log(f"开始处理: {len(image_paths)} 张")
        self._log(f"方法: {method_name}")
        self._log(f"极角采样数: {num_angles}")
        if center_xy is None:
            self._log("中心: 自动")
        else:
            self._log(f"中心: 手动 (x={center_xy[0]:.2f}, y={center_xy[1]:.2f})")

        started = time.perf_counter()
        success_count = 0
        failed: List[Tuple[str, str]] = []

        for idx, image_path in enumerate(image_paths, start=1):
            base_name = os.path.basename(image_path)
            try:
                image = self._load_image(image_path)
                result = self._process_single(image, center_xy, num_angles)
                save_path = self._build_output_path(base_name)
                tiff.imwrite(save_path, result.astype(np.float32), dtype=np.float32)

                success_count += 1
                self._log(
                    f"[{idx}/{len(image_paths)}] 成功: {base_name} -> {os.path.basename(save_path)}"
                )
            except Exception as e:
                failed.append((base_name, str(e)))
                self._log(f"[{idx}/{len(image_paths)}] 失败: {base_name}, 原因: {e}")

            QApplication.processEvents()

        elapsed = time.perf_counter() - started
        self._log("-" * 60)
        self._log(
            f"处理完成: 成功 {success_count}, 失败 {len(failed)}, 总耗时 {elapsed:.2f}s"
        )
        self._log(f"输出目录: {self.output_folder}")

        if failed:
            self._log("失败文件列表:")
            for name, reason in failed:
                self._log(f"  - {name}: {reason}")

        QMessageBox.information(
            self,
            "完成",
            f"处理完成。\n成功: {success_count}\n失败: {len(failed)}\n耗时: {elapsed:.2f}s",
        )

    def _process_single(
        self,
        image: np.ndarray,
        center_xy: Optional[Tuple[float, float]],
        num_angles: int,
    ) -> np.ndarray:
        method_idx = self.method_combo.currentIndex()

        if method_idx == 0:
            return remove_ring_artifact_polar(
                image,
                center_xy=center_xy,
                num_angles=num_angles,
                stripe_sigma=self.polar_sigma_spin.value(),
                correction_strength=self.polar_strength_spin.value(),
                transformer=self.transformer,
            )

        if method_idx == 1:
            return remove_ring_artifact_frequency(
                image,
                center_xy=center_xy,
                num_angles=num_angles,
                low_freq_cutoff=self.freq_cutoff_spin.value(),
                suppression_ratio=self.freq_suppression_spin.value(),
                periodic_notch=self.freq_notch_spin.value(),
                notch_width=self.freq_notch_width_spin.value(),
                transformer=self.transformer,
            )

        return remove_ring_artifact_morphology(
            image,
            center_xy=center_xy,
            num_angles=num_angles,
            opening_theta_size=self.morph_opening_spin.value(),
            profile_sigma=self.morph_sigma_spin.value(),
            correction_strength=self.morph_strength_spin.value(),
            transformer=self.transformer,
        )

    def _collect_image_paths(self, folder: str) -> List[str]:
        valid_exts = (".tif", ".tiff", ".raw")
        files = [
            os.path.join(folder, file_name)
            for file_name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, file_name))
            and file_name.lower().endswith(valid_exts)
        ]
        files.sort(key=lambda p: os.path.basename(p).lower())
        return files

    def _load_image(self, image_path: str) -> np.ndarray:
        lower_name = image_path.lower()
        if lower_name.endswith(".raw"):
            width = int(self.raw_width_spin.value())
            height = int(self.raw_height_spin.value())
            data = np.fromfile(image_path, dtype=np.uint16)
            expected = width * height
            if data.size != expected:
                raise ValueError(
                    f"RAW 尺寸不匹配: 设置 {width}x{height}={expected}, 文件像素 {data.size}"
                )
            image = data.reshape((height, width))
        else:
            image = np.array(Image.open(image_path))
            if image.ndim == 3:
                image = image[:, :, 0]

        return image.astype(np.float32, copy=False)

    def _build_output_path(self, base_name: str) -> str:
        stem, _ = os.path.splitext(base_name)
        return os.path.join(self.output_folder, f"{stem}.tif")

    def _get_center_xy(self) -> Optional[Tuple[float, float]]:
        if self.center_mode_combo.currentIndex() == 0:
            return None
        return (self.center_x_spin.value(), self.center_y_spin.value())

    def _resolve_template_store_path(self) -> str:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        return os.path.join(
            project_root, ".trae", "config", "ring_artifact_templates.json"
        )

    def _load_custom_templates(self):
        self.custom_templates = {}
        if not os.path.exists(self.template_store_path):
            return

        try:
            with open(self.template_store_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and isinstance(data.get("templates"), dict):
                self.custom_templates = data["templates"]
            elif isinstance(data, dict):
                self.custom_templates = data
            else:
                self._log("模板文件格式无效，已忽略。")
        except Exception as e:
            self._log(f"读取模板失败: {e}")

    def _save_custom_templates(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.template_store_path), exist_ok=True)
            payload = {
                "templates": self.custom_templates,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.template_store_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self._log(f"保存模板失败: {e}")
            return False

    def _refresh_template_combo(self, selected_name: Optional[str] = None):
        names = sorted(self.custom_templates.keys())
        self.template_combo.clear()
        self.template_combo.addItems(names)

        if not names:
            return

        if selected_name and selected_name in names:
            self.template_combo.setCurrentText(selected_name)
        else:
            self.template_combo.setCurrentIndex(0)

    def _collect_template_payload(self) -> dict:
        return {
            "raw_width": int(self.raw_width_spin.value()),
            "raw_height": int(self.raw_height_spin.value()),
            "center_mode": int(self.center_mode_combo.currentIndex()),
            "center_x": float(self.center_x_spin.value()),
            "center_y": float(self.center_y_spin.value()),
            "method": int(self.method_combo.currentIndex()),
            "num_angles": int(self.num_angles_spin.value()),
            "polar_sigma": float(self.polar_sigma_spin.value()),
            "polar_strength": float(self.polar_strength_spin.value()),
            "freq_cutoff": int(self.freq_cutoff_spin.value()),
            "freq_suppression": float(self.freq_suppression_spin.value()),
            "freq_notch": int(self.freq_notch_spin.value()),
            "freq_notch_width": int(self.freq_notch_width_spin.value()),
            "morph_opening": int(self.morph_opening_spin.value()),
            "morph_sigma": float(self.morph_sigma_spin.value()),
            "morph_strength": float(self.morph_strength_spin.value()),
        }

    def _apply_template_payload(self, payload: dict):
        self.raw_width_spin.setValue(
            int(payload.get("raw_width", self.raw_width_spin.value()))
        )
        self.raw_height_spin.setValue(
            int(payload.get("raw_height", self.raw_height_spin.value()))
        )

        center_mode = int(
            payload.get("center_mode", self.center_mode_combo.currentIndex())
        )
        if center_mode not in (0, 1):
            center_mode = 0
        self.center_mode_combo.setCurrentIndex(center_mode)
        self._on_center_mode_changed(center_mode)
        self.center_x_spin.setValue(
            float(payload.get("center_x", self.center_x_spin.value()))
        )
        self.center_y_spin.setValue(
            float(payload.get("center_y", self.center_y_spin.value()))
        )

        method_idx = int(payload.get("method", self.method_combo.currentIndex()))
        method_idx = max(0, min(2, method_idx))
        self.method_combo.setCurrentIndex(method_idx)
        self._on_method_changed(method_idx)

        self.num_angles_spin.setValue(
            int(payload.get("num_angles", self.num_angles_spin.value()))
        )

        self.polar_sigma_spin.setValue(
            float(payload.get("polar_sigma", self.polar_sigma_spin.value()))
        )
        self.polar_strength_spin.setValue(
            float(payload.get("polar_strength", self.polar_strength_spin.value()))
        )

        self.freq_cutoff_spin.setValue(
            int(payload.get("freq_cutoff", self.freq_cutoff_spin.value()))
        )
        self.freq_suppression_spin.setValue(
            float(payload.get("freq_suppression", self.freq_suppression_spin.value()))
        )
        self.freq_notch_spin.setValue(
            int(payload.get("freq_notch", self.freq_notch_spin.value()))
        )
        self.freq_notch_width_spin.setValue(
            int(payload.get("freq_notch_width", self.freq_notch_width_spin.value()))
        )

        self.morph_opening_spin.setValue(
            int(payload.get("morph_opening", self.morph_opening_spin.value()))
        )
        self.morph_sigma_spin.setValue(
            float(payload.get("morph_sigma", self.morph_sigma_spin.value()))
        )
        self.morph_strength_spin.setValue(
            float(payload.get("morph_strength", self.morph_strength_spin.value()))
        )

    def _log(self, message: str):
        self.log_text.append(message)
