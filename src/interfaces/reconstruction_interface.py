import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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

from src.interfaces.ui_theme import apply_interface_theme, set_button_role
from src.reconstruction.fdk_runner import (
    ReconstructionPreviewResult,
    ReconstructionRunResult,
    run_fdk_reconstruction,
    run_reconstruction_preview_slice,
)
from src.reconstruction.io_loader import collect_projection_files
from src.reconstruction.models import ReconstructionConfig
from src.reconstruction.pipeline import build_stage1_plan, validate_config


class ReconstructionInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("reconstructionInterface")
        self.setWindowTitle("CBCT 重构")
        self.resize(1600, 900)
        apply_interface_theme(self)
        self._stop_requested = False
        self._worker_process = None
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
        geometry_group = self._build_geometry_group()
        recon_group = self._build_recon_group()
        derived_group = self._build_derived_group()
        action_group = self._build_action_group()

        note_label = QLabel(
            "阶段3说明：本页面支持 CBCT 重构参数录入、配置校验与重构执行。\n"
            "当前默认角度规则：theta_i = 起始角度 + i * 每帧间隔角度。\n"
            "当前默认体尺寸规则：X=Y=投影宽, Z=投影高。\n"
            "已接入：子进程重构执行（可指定 astra_env 解释器）+ FDK/SIRT/CGLS。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")

        controls_layout.addWidget(folder_group, 0, 0, 1, 2)
        controls_layout.addWidget(geometry_group, 1, 0)
        controls_layout.addWidget(recon_group, 1, 1)
        controls_layout.addWidget(derived_group, 2, 0)
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

        self._refresh_projection_count()
        self._on_auto_volume_toggled()
        self._update_derived_fields()
        self._on_algorithm_changed()

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox("数据路径")
        layout = QVBoxLayout()

        self.projection_edit = self._build_folder_row(
            layout,
            "投影目录",
            self.on_select_projection_folder,
        )
        self.output_edit = self._build_folder_row(
            layout,
            "输出目录",
            self.on_select_output_folder,
        )

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("投影数量"))
        self.projection_count_value = QLineEdit("0")
        self.projection_count_value.setReadOnly(True)
        count_row.addWidget(self.projection_count_value)
        layout.addLayout(count_row)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("投影尺寸 (H x W)"))
        self.projection_shape_value = QLineEdit("未知")
        self.projection_shape_value.setReadOnly(True)
        shape_row.addWidget(self.projection_shape_value)
        layout.addLayout(shape_row)

        group.setLayout(layout)
        return group

    def _build_geometry_group(self) -> QGroupBox:
        group = QGroupBox("几何参数")
        form = QFormLayout()

        self.sod_spin = QDoubleSpinBox()
        self.sod_spin.setRange(1e-6, 1e9)
        self.sod_spin.setDecimals(6)
        self.sod_spin.setValue(1000.0)
        self.sod_spin.valueChanged.connect(self._update_derived_fields)
        form.addRow("SOD (mm)", self.sod_spin)

        self.sdd_spin = QDoubleSpinBox()
        self.sdd_spin.setRange(1e-6, 1e9)
        self.sdd_spin.setDecimals(6)
        self.sdd_spin.setValue(1500.0)
        self.sdd_spin.valueChanged.connect(self._update_derived_fields)
        form.addRow("SDD (mm)", self.sdd_spin)

        self.angle_step_spin = QDoubleSpinBox()
        self.angle_step_spin.setRange(-360.0, 360.0)
        self.angle_step_spin.setDecimals(8)
        self.angle_step_spin.setValue(0.3)
        self.angle_step_spin.valueChanged.connect(self._update_derived_fields)
        form.addRow("每帧角度间隔 (deg)", self.angle_step_spin)

        self.start_angle_spin = QDoubleSpinBox()
        self.start_angle_spin.setRange(-100000.0, 100000.0)
        self.start_angle_spin.setDecimals(8)
        self.start_angle_spin.setValue(0.0)
        form.addRow("起始角度 (deg)", self.start_angle_spin)

        self.pixel_x_spin = QDoubleSpinBox()
        self.pixel_x_spin.setRange(1e-9, 1e6)
        self.pixel_x_spin.setDecimals(9)
        self.pixel_x_spin.setValue(0.139)
        self.pixel_x_spin.valueChanged.connect(self._update_derived_fields)
        form.addRow("探测器像素尺寸 X (mm)", self.pixel_x_spin)

        self.pixel_y_spin = QDoubleSpinBox()
        self.pixel_y_spin.setRange(1e-9, 1e6)
        self.pixel_y_spin.setDecimals(9)
        self.pixel_y_spin.setValue(0.139)
        self.pixel_y_spin.valueChanged.connect(self._update_derived_fields)
        form.addRow("探测器像素尺寸 Y (mm)", self.pixel_y_spin)

        self.cor_offset_spin = QDoubleSpinBox()
        self.cor_offset_spin.setRange(-10000.0, 10000.0)
        self.cor_offset_spin.setDecimals(6)
        self.cor_offset_spin.setValue(0.0)
        form.addRow("COR 轴偏移 (pixel)", self.cor_offset_spin)

        cor_search_widget = QWidget()
        cor_search_layout = QHBoxLayout(cor_search_widget)
        cor_search_layout.setContentsMargins(0, 0, 0, 0)

        self.cor_search_btn = QPushButton("自动搜索COR")
        self.cor_search_btn.clicked.connect(self.on_auto_search_cor_clicked)
        cor_search_layout.addWidget(self.cor_search_btn)

        cor_search_layout.addWidget(QLabel("范围±"))
        self.cor_search_range_spin = QSpinBox()
        self.cor_search_range_spin.setRange(1, 2000)
        self.cor_search_range_spin.setValue(80)
        cor_search_layout.addWidget(self.cor_search_range_spin)
        cor_search_layout.addWidget(QLabel("px"))

        cor_search_layout.addWidget(QLabel("降采样"))
        self.cor_search_downsample_spin = QSpinBox()
        self.cor_search_downsample_spin.setRange(1, 16)
        self.cor_search_downsample_spin.setValue(4)
        cor_search_layout.addWidget(self.cor_search_downsample_spin)
        cor_search_layout.addStretch(1)
        form.addRow("COR 自动搜索", cor_search_widget)

        group.setLayout(form)
        return group

    def _build_recon_group(self) -> QGroupBox:
        group = QGroupBox("重构参数")
        form = QFormLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setProperty("_allow_wheel_change", False)
        self.algorithm_combo.setFocusPolicy(Qt.StrongFocus)
        self.algorithm_combo.addItems(
            [
                "FDK (Cone-beam)",
                "SIRT3D_CUDA",
                "CGLS3D_CUDA",
                "FDK + CGLS3D_CUDA",
            ]
        )
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        form.addRow("重构算法", self.algorithm_combo)

        self.filter_combo = QComboBox()
        self.filter_combo.setProperty("_allow_wheel_change", False)
        self.filter_combo.setFocusPolicy(Qt.StrongFocus)
        self.filter_combo.addItems(["Ram-Lak", "Shepp-Logan", "Hann", "Hamming"])
        form.addRow("滤波器", self.filter_combo)

        self.auto_volume_checkbox = QCheckBox("按投影尺寸自动设置 XYZ（推荐）")
        self.auto_volume_checkbox.setChecked(True)
        self.auto_volume_checkbox.stateChanged.connect(self._on_auto_volume_toggled)
        form.addRow("体尺寸策略", self.auto_volume_checkbox)

        self.recon_nx_spin = QSpinBox()
        self.recon_nx_spin.setRange(1, 16384)
        self.recon_nx_spin.setValue(1)
        form.addRow("重构体尺寸 X", self.recon_nx_spin)

        self.recon_ny_spin = QSpinBox()
        self.recon_ny_spin.setRange(1, 16384)
        self.recon_ny_spin.setValue(1)
        form.addRow("重构体尺寸 Y", self.recon_ny_spin)

        self.recon_nz_spin = QSpinBox()
        self.recon_nz_spin.setRange(1, 16384)
        self.recon_nz_spin.setValue(1)
        form.addRow("重构体尺寸 Z", self.recon_nz_spin)

        self.iterative_iter_spin = QSpinBox()
        self.iterative_iter_spin.setRange(1, 1000)
        self.iterative_iter_spin.setValue(30)
        self.iterative_iter_spin.setEnabled(False)
        form.addRow("迭代重构次数", self.iterative_iter_spin)

        self.refine_iter_spin = QSpinBox()
        self.refine_iter_spin.setRange(0, 50)
        self.refine_iter_spin.setValue(0)
        form.addRow("后处理扩散次数", self.refine_iter_spin)

        self.refine_step_spin = QDoubleSpinBox()
        self.refine_step_spin.setRange(0.0, 0.24)
        self.refine_step_spin.setDecimals(5)
        self.refine_step_spin.setSingleStep(0.01)
        self.refine_step_spin.setValue(0.08)
        form.addRow("后处理扩散步长", self.refine_step_spin)

        self.output_format_combo = QComboBox()
        self.output_format_combo.setProperty("_allow_wheel_change", False)
        self.output_format_combo.setFocusPolicy(Qt.StrongFocus)
        self.output_format_combo.addItems(["TIFF 切片"])
        form.addRow("输出格式", self.output_format_combo)

        group.setLayout(form)
        return group

    def _build_derived_group(self) -> QGroupBox:
        group = QGroupBox("自动计算（只读）")
        form = QFormLayout()

        self.magnification_edit = QLineEdit()
        self.magnification_edit.setReadOnly(True)
        form.addRow("放大倍数 M = SDD / SOD", self.magnification_edit)

        self.voxel_x_edit = QLineEdit()
        self.voxel_x_edit.setReadOnly(True)
        form.addRow("体素尺寸 X (mm)", self.voxel_x_edit)

        self.voxel_y_edit = QLineEdit()
        self.voxel_y_edit.setReadOnly(True)
        form.addRow("体素尺寸 Y (mm)", self.voxel_y_edit)

        self.voxel_z_edit = QLineEdit()
        self.voxel_z_edit.setReadOnly(True)
        form.addRow("体素尺寸 Z (mm)", self.voxel_z_edit)

        self.total_angle_edit = QLineEdit()
        self.total_angle_edit.setReadOnly(True)
        form.addRow("总扫描角度 (deg)", self.total_angle_edit)

        group.setLayout(form)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("执行")
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.use_subprocess_checkbox = QCheckBox("使用子进程执行（推荐）")
        self.use_subprocess_checkbox.setChecked(True)
        layout.addWidget(self.use_subprocess_checkbox)

        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel("Python解释器"))
        self.worker_python_edit = QLineEdit(sys.executable)
        runtime_row.addWidget(self.worker_python_edit, 1)
        self.worker_python_btn = QPushButton("浏览")
        self.worker_python_btn.clicked.connect(self.on_select_worker_python)
        runtime_row.addWidget(self.worker_python_btn)
        layout.addLayout(runtime_row)

        btn_row = QHBoxLayout()
        self.validate_btn = QPushButton("校验参数")
        set_button_role(self.validate_btn, "primary")
        self.validate_btn.clicked.connect(self.on_validate_clicked)
        btn_row.addWidget(self.validate_btn)

        self.run_btn = QPushButton("开始重构（阶段3）")
        set_button_role(self.run_btn, "primary")
        self.run_btn.clicked.connect(self.on_run_clicked)
        btn_row.addWidget(self.run_btn)

        self.preview_btn = QPushButton("预览中间切片")
        set_button_role(self.preview_btn, "primary")
        self.preview_btn.clicked.connect(self.on_preview_middle_clicked)
        btn_row.addWidget(self.preview_btn)

        self.stop_btn = QPushButton("停止")
        set_button_role(self.stop_btn, "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.clear_log_btn = QPushButton("清空日志")
        set_button_role(self.clear_log_btn, "danger")
        self.clear_log_btn.clicked.connect(self._clear_log)
        layout.addWidget(self.clear_log_btn)

        group.setLayout(layout)
        return group

    def _build_folder_row(
        self, parent_layout: QVBoxLayout, title: str, slot
    ) -> QLineEdit:
        row = QHBoxLayout()
        label = QLabel(title)
        edit = QLineEdit()
        button = QPushButton("浏览")
        button.clicked.connect(slot)
        row.addWidget(label)
        row.addWidget(edit, 1)
        row.addWidget(button)
        parent_layout.addLayout(row)
        return edit

    def _select_folder(self, title: str, target: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, title, "")
        if folder:
            target.setText(folder)
            self._log(f"{title}: {folder}")
            if target is self.projection_edit:
                self._refresh_projection_count()
                if (
                    self.auto_volume_checkbox.isChecked()
                    and self.projection_shape_value.text().strip() == "未知"
                ):
                    self._log(
                        "未能自动读取投影尺寸（例如 RAW 无尺寸信息），"
                        "请手动填写重构体尺寸 XYZ。"
                    )

    def on_select_projection_folder(self):
        self._select_folder("选择投影目录", self.projection_edit)

    def on_select_output_folder(self):
        self._select_folder("选择输出目录", self.output_edit)

    def _on_algorithm_changed(self, *_):
        algo_text = self.algorithm_combo.currentText().strip().lower()
        iterative_mode = ("sirt" in algo_text) or ("cgls" in algo_text)
        self.iterative_iter_spin.setEnabled(iterative_mode)

    def _prepare_projection_for_cor_search(
        self, image: np.ndarray, downsample: int
    ) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if downsample > 1:
            arr = arr[::downsample, ::downsample]

        h = int(arr.shape[0])
        y0 = int(round(h * 0.2))
        y1 = int(round(h * 0.8))
        if y1 <= y0:
            y0 = 0
            y1 = h
        arr = arr[y0:y1, :]

        row_mean = arr.mean(axis=1, keepdims=True)
        row_std = arr.std(axis=1, keepdims=True) + 1e-6
        arr = (arr - row_mean) / row_std
        return arr.astype(np.float32, copy=False)

    def _mirror_shift_l1_score(
        self, a: np.ndarray, b_flip: np.ndarray, shift: int
    ) -> float:
        h, w = a.shape
        if b_flip.shape != (h, w):
            raise ValueError("COR 搜索图像尺寸不一致。")

        if shift >= 0:
            if shift >= w - 8:
                return float("inf")
            x = a[:, shift:]
            y = b_flip[:, : w - shift]
        else:
            if -shift >= w - 8:
                return float("inf")
            x = a[:, : w + shift]
            y = b_flip[:, -shift:]

        if x.size == 0 or x.shape[1] < 8:
            return float("inf")
        return float(np.mean(np.abs(x - y)))

    def _estimate_cor_from_pair(
        self,
        proj_a: np.ndarray,
        proj_b: np.ndarray,
        search_range_px: int,
        downsample: int,
    ) -> Tuple[float, float]:
        a = self._prepare_projection_for_cor_search(proj_a, downsample)
        b = self._prepare_projection_for_cor_search(proj_b, downsample)
        b_flip = np.flip(b, axis=1)

        max_shift_ds = max(
            1, int(np.ceil((2.0 * float(search_range_px)) / float(downsample)))
        )
        shifts = np.arange(-max_shift_ds, max_shift_ds + 1, dtype=np.int32)
        scores = np.empty(shifts.shape[0], dtype=np.float64)

        for idx, shift in enumerate(shifts.tolist()):
            scores[idx] = self._mirror_shift_l1_score(a, b_flip, int(shift))

        best_idx = int(np.argmin(scores))
        best_shift = float(shifts[best_idx])

        # Quadratic refinement around best integer shift.
        if 0 < best_idx < (len(shifts) - 1):
            y_m1 = float(scores[best_idx - 1])
            y_0 = float(scores[best_idx])
            y_p1 = float(scores[best_idx + 1])
            denom = y_m1 - 2.0 * y_0 + y_p1
            if abs(denom) > 1e-12:
                delta = 0.5 * (y_m1 - y_p1) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                best_shift = best_shift + delta

        # Our shift score moves mirrored image; COR is opposite sign and half shift.
        cor_px = -0.5 * best_shift * float(downsample)
        best_score = float(scores[best_idx])
        return float(cor_px), best_score

    def on_auto_search_cor_clicked(self):
        folder = self.projection_edit.text().strip()
        projection_files = collect_projection_files(folder) if folder else []
        if len(projection_files) < 2:
            QMessageBox.warning(self, "提示", "请先选择包含足够投影图像的目录。")
            return

        angle_step = abs(float(self.angle_step_spin.value()))
        if angle_step < 1e-8:
            QMessageBox.warning(self, "提示", "每帧角度间隔不能为 0。")
            return

        total_angle = (len(projection_files) - 1) * angle_step
        if total_angle < 170.0:
            QMessageBox.warning(
                self,
                "提示",
                "当前投影总角度不足约 180°，无法可靠执行 0/180 配对的 COR 自动搜索。",
            )
            return

        pair_offset = int(round(180.0 / angle_step))
        if pair_offset <= 0 or pair_offset >= len(projection_files):
            pair_offset = len(projection_files) // 2
        if pair_offset <= 0 or pair_offset >= len(projection_files):
            QMessageBox.warning(
                self, "提示", "无法构建 COR 搜索配对，请检查投影数量和角度设置。"
            )
            return

        usable_count = len(projection_files) - pair_offset
        num_pairs = int(min(5, max(1, usable_count)))
        pair_indices = np.linspace(
            0, usable_count - 1, num=num_pairs, dtype=np.int32
        ).tolist()

        search_range = int(self.cor_search_range_spin.value())
        downsample = int(self.cor_search_downsample_spin.value())
        self._log("-" * 60)
        self._log(
            f"开始自动搜索 COR: 配对偏移={pair_offset}, 配对数={num_pairs}, "
            f"范围=±{search_range}px, 降采样={downsample}"
        )

        cor_estimates: List[float] = []
        for idx in pair_indices:
            j = idx + pair_offset
            if j >= len(projection_files):
                continue

            try:
                proj_a = iio.imread(projection_files[idx])
                proj_b = iio.imread(projection_files[j])
                cor_px, score = self._estimate_cor_from_pair(
                    proj_a=proj_a,
                    proj_b=proj_b,
                    search_range_px=search_range,
                    downsample=downsample,
                )
                cor_estimates.append(float(cor_px))
                self._log(
                    f"COR 搜索配对 {idx}-{j}: cor={cor_px:.3f} px, score={score:.6f}"
                )
            except Exception as e:
                self._log(f"COR 搜索配对 {idx}-{j} 失败: {e}")
            QApplication.processEvents()

        if not cor_estimates:
            QMessageBox.warning(self, "提示", "COR 自动搜索失败，未得到有效估计。")
            return

        cor_array = np.asarray(cor_estimates, dtype=np.float64)
        median = float(np.median(cor_array))
        mad = float(np.median(np.abs(cor_array - median)))
        if mad > 1e-6 and cor_array.size >= 3:
            keep = np.abs(cor_array - median) <= (2.5 * mad)
            if np.any(keep):
                cor_final = float(np.mean(cor_array[keep]))
            else:
                cor_final = median
        else:
            cor_final = median

        self.cor_offset_spin.setValue(cor_final)
        self._log(
            f"自动搜索 COR 完成: 推荐值 {cor_final:.3f} px "
            f"(median={median:.3f}, mad={mad:.3f})"
        )
        self._log("你可以在此基础上做小范围微调。")

    def on_select_worker_python(self):
        start_dir = self.worker_python_edit.text().strip() or ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Python 解释器",
            start_dir,
            "Python (*.exe);;All Files (*)",
        )
        if file_path:
            self.worker_python_edit.setText(file_path)
            self._log(f"已设置 Python 解释器: {file_path}")

    def _project_root(self) -> str:
        return str(Path(__file__).resolve().parents[2])

    def _handle_worker_stdout_line(self, line: str, on_progress):
        text = (line or "").strip()
        if not text:
            return
        if text.startswith("PROGRESS\t"):
            parts = text.split("\t", 3)
            if len(parts) >= 4:
                try:
                    done = int(parts[1])
                    total = int(parts[2])
                    message = parts[3]
                    on_progress(done, total, message)
                    return
                except Exception:
                    pass
        if text.startswith("ERROR\t"):
            self._log(text.replace("ERROR\t", "子进程错误: ", 1))
            return
        if text.startswith("RESULT\t"):
            self._log(text.replace("RESULT\t", "子进程结果: ", 1))
            return
        self._log(f"[worker] {text}")

    def _run_reconstruction_subprocess(
        self,
        config: ReconstructionConfig,
        projection_files: List[str],
        on_progress,
    ) -> ReconstructionRunResult:
        tmp_root = os.path.join(config.output_folder, "_recon_job")
        os.makedirs(tmp_root, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="job_", dir=tmp_root)
        config_json_path = os.path.join(tmp_dir, "config.json")
        projection_json_path = os.path.join(tmp_dir, "projection_files.json")
        result_json_path = os.path.join(tmp_dir, "result.json")

        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f, ensure_ascii=False, indent=2)
        with open(projection_json_path, "w", encoding="utf-8") as f:
            json.dump(projection_files, f, ensure_ascii=False, indent=2)

        python_exec = self.worker_python_edit.text().strip() or sys.executable
        cmd = [
            python_exec,
            "-m",
            "src.reconstruction.recon_worker",
            "--config-json",
            config_json_path,
            "--projection-list-json",
            projection_json_path,
            "--result-json",
            result_json_path,
        ]

        self._log(f"阶段3子进程命令: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            cwd=self._project_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._worker_process = process
        line_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        stderr_lines: List[str] = []

        def _reader(src: str, pipe):
            try:
                for line in iter(pipe.readline, ""):
                    line_queue.put((src, line.rstrip("\r\n")))
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        stdout_thread = threading.Thread(
            target=_reader, args=("stdout", process.stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_reader, args=("stderr", process.stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            while True:
                drained = False
                while True:
                    try:
                        src, line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    if src == "stdout":
                        self._handle_worker_stdout_line(line, on_progress)
                    else:
                        if line.strip():
                            stderr_lines.append(line.strip())
                            self._log(f"[worker][stderr] {line.strip()}")

                if self._stop_requested and process.poll() is None:
                    self._log("检测到停止请求，正在终止阶段3子进程...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError("用户中止重构。")

                if process.poll() is not None:
                    break

                if not drained:
                    QApplication.processEvents()
                    time.sleep(0.05)

            # Drain remaining lines after process exits.
            while True:
                try:
                    src, line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if src == "stdout":
                    self._handle_worker_stdout_line(line, on_progress)
                else:
                    if line.strip():
                        stderr_lines.append(line.strip())
                        self._log(f"[worker][stderr] {line.strip()}")

            return_code = process.returncode
            if return_code != 0:
                stderr_tail = "\n".join(stderr_lines[-20:]).strip()
                if not stderr_tail:
                    stderr_tail = "子进程异常退出，但没有可用 stderr 输出。"
                raise RuntimeError(
                    f"阶段3子进程重构失败 (exit={return_code})\n{stderr_tail}"
                )

            if not os.path.exists(result_json_path):
                raise RuntimeError("阶段3子进程未生成 result.json。")

            with open(result_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return ReconstructionRunResult(
                output_slice_dir=str(payload.get("output_slice_dir", "")),
                summary_json_path=str(payload.get("summary_json_path", "")),
                angles_csv_path=str(payload.get("angles_csv_path", "")),
                stage1_plan_json_path=str(payload.get("stage1_plan_json_path", "")),
                slice_count=int(payload.get("slice_count", 0)),
                projection_count=int(payload.get("projection_count", 0)),
                elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
                backend_used=str(payload.get("backend_used", "unknown")),
                backend_note=str(payload.get("backend_note", "")),
            )
        finally:
            self._worker_process = None

    def _refresh_projection_count(self) -> int:
        folder = self.projection_edit.text().strip()
        files = collect_projection_files(folder) if folder else []
        count = len(files)
        self.projection_count_value.setText(str(count))
        shape = self._detect_projection_shape(files)
        if shape is None:
            self.projection_shape_value.setText("未知")
        else:
            self.projection_shape_value.setText(f"{shape[0]} x {shape[1]}")
        self._apply_volume_dims_from_projection_shape(shape, log_change=False)
        self._on_auto_volume_toggled()
        self._update_derived_fields()
        return count

    def _detect_projection_shape(
        self, projection_files: List[str]
    ) -> Optional[Tuple[int, int]]:
        if not projection_files:
            return None
        first_path = projection_files[0]
        if first_path.lower().endswith(".raw"):
            return None
        try:
            image = iio.imread(first_path)
        except Exception:
            return None
        if image.ndim == 3:
            image = image[:, :, 0]
        if image.ndim != 2:
            return None
        return int(image.shape[0]), int(image.shape[1])

    def _apply_volume_dims_from_projection_shape(
        self,
        shape: Optional[Tuple[int, int]],
        log_change: bool,
    ):
        auto_enabled = (
            hasattr(self, "auto_volume_checkbox")
            and self.auto_volume_checkbox.isChecked()
        )
        if not auto_enabled or shape is None:
            return
        proj_h, proj_w = shape
        nx = int(proj_w)
        ny = int(proj_w)
        nz = int(proj_h)
        self.recon_nx_spin.setValue(nx)
        self.recon_ny_spin.setValue(ny)
        self.recon_nz_spin.setValue(nz)
        if log_change:
            self._log(
                "已按投影尺寸自动设置重构体尺寸: "
                f"X={nx}, Y={ny}, Z={nz} "
                "(规则: X=Y=投影宽, Z=投影高)"
            )

    def _on_auto_volume_toggled(self):
        auto_enabled = self.auto_volume_checkbox.isChecked()
        self.recon_nx_spin.setEnabled(True)
        self.recon_ny_spin.setEnabled(True)
        self.recon_nz_spin.setEnabled(True)
        if auto_enabled:
            shape_text = self.projection_shape_value.text().strip()
            shape = None
            if "x" in shape_text:
                try:
                    left, right = shape_text.split("x")
                    shape = (int(left.strip()), int(right.strip()))
                except Exception:
                    shape = None
            if shape is not None:
                self._apply_volume_dims_from_projection_shape(shape, log_change=False)
                self.recon_nx_spin.setEnabled(False)
                self.recon_ny_spin.setEnabled(False)
                self.recon_nz_spin.setEnabled(False)

    def _build_config_and_files(
        self,
    ) -> Tuple[Optional[ReconstructionConfig], List[str]]:
        projection_folder = self.projection_edit.text().strip()
        output_folder = self.output_edit.text().strip()
        if not projection_folder:
            QMessageBox.warning(self, "警告", "请先选择投影目录。")
            return None, []
        if not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出目录。")
            return None, []

        projection_files = collect_projection_files(projection_folder)
        if not projection_files:
            QMessageBox.warning(self, "警告", "投影目录中未找到可处理图像文件。")
            return None, []

        config = ReconstructionConfig(
            projection_folder=projection_folder,
            output_folder=output_folder,
            projection_count=len(projection_files),
            sod_mm=float(self.sod_spin.value()),
            sdd_mm=float(self.sdd_spin.value()),
            angle_step_deg=float(self.angle_step_spin.value()),
            start_angle_deg=float(self.start_angle_spin.value()),
            detector_pixel_size_x_mm=float(self.pixel_x_spin.value()),
            detector_pixel_size_y_mm=float(self.pixel_y_spin.value()),
            cor_offset_px=float(self.cor_offset_spin.value()),
            algorithm=self.algorithm_combo.currentText(),
            iterative_iterations=(
                int(self.iterative_iter_spin.value())
                if self.iterative_iter_spin.isEnabled()
                else 0
            ),
            filter_name=self.filter_combo.currentText(),
            recon_nx=int(self.recon_nx_spin.value()),
            recon_ny=int(self.recon_ny_spin.value()),
            recon_nz=int(self.recon_nz_spin.value()),
            output_format=self.output_format_combo.currentText(),
            refine_iterations=int(self.refine_iter_spin.value()),
            refine_step=float(self.refine_step_spin.value()),
        )
        return config, projection_files

    def _update_derived_fields(self):
        count = int(self.projection_count_value.text().strip() or "0")
        sod = float(self.sod_spin.value())
        sdd = float(self.sdd_spin.value())
        pixel_x = float(self.pixel_x_spin.value())
        pixel_y = float(self.pixel_y_spin.value())
        angle_step = float(self.angle_step_spin.value())

        if sod <= 0 or sdd <= 0:
            self.magnification_edit.setText("N/A")
            self.voxel_x_edit.setText("N/A")
            self.voxel_y_edit.setText("N/A")
            self.voxel_z_edit.setText("N/A")
            self.total_angle_edit.setText("N/A")
            return

        magnification = sdd / sod
        if magnification <= 0:
            self.magnification_edit.setText("N/A")
            self.voxel_x_edit.setText("N/A")
            self.voxel_y_edit.setText("N/A")
            self.voxel_z_edit.setText("N/A")
            self.total_angle_edit.setText("N/A")
            return

        voxel_x = pixel_x / magnification
        voxel_y = pixel_y / magnification
        voxel_z = voxel_y
        total_angle = (count - 1) * angle_step if count > 0 else 0.0

        self.magnification_edit.setText(f"{magnification:.8f}")
        self.voxel_x_edit.setText(f"{voxel_x:.8f}")
        self.voxel_y_edit.setText(f"{voxel_y:.8f}")
        self.voxel_z_edit.setText(f"{voxel_z:.8f}")
        self.total_angle_edit.setText(f"{total_angle:.8f}")

    def on_validate_clicked(self):
        self.progress_bar.setValue(0)
        config, projection_files = self._build_config_and_files()
        if config is None:
            return
        try:
            validate_config(config)
            derived, angles = build_stage1_plan(config)
            self._log("-" * 60)
            self._log("参数校验通过。")
            self._log(f"投影数量: {len(projection_files)}")
            self._log(f"重构算法: {config.algorithm}")
            if config.iterative_iterations > 0:
                self._log(f"迭代重构次数: {config.iterative_iterations}")
            shape = self._detect_projection_shape(projection_files)
            if shape is not None:
                self._log(f"投影尺寸: H={shape[0]}, W={shape[1]}")
                self._log(
                    "重构体尺寸规则: X=Y=投影宽, Z=投影高 | "
                    f"当前 X={config.recon_nx}, Y={config.recon_ny}, Z={config.recon_nz}"
                )
            self._log(
                f"放大倍数: {derived.magnification:.8f} | "
                f"体素(mm): x={derived.voxel_size_x_mm:.8f}, "
                f"y={derived.voxel_size_y_mm:.8f}, z={derived.voxel_size_z_mm:.8f}"
            )
            self._log(
                f"角度范围: {float(angles[0]):.8f} -> {float(angles[-1]):.8f} "
                f"(总角度 {derived.total_scan_angle_deg:.8f})"
            )
            self.progress_bar.setValue(100)
        except Exception as e:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "校验失败", str(e))
            self._log(f"校验失败: {e}")

    def on_run_clicked(self):
        self.progress_bar.setValue(0)
        config, projection_files = self._build_config_and_files()
        if config is None:
            return

        self._stop_requested = False
        self.run_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.clear_log_btn.setEnabled(False)

        self._log("-" * 60)
        self._log("开始阶段3重构。")
        self._log(f"投影数量: {len(projection_files)}")
        self._log(
            "执行模式: 子进程"
            if self.use_subprocess_checkbox.isChecked()
            else "执行模式: 当前进程"
        )
        self._log(f"重构算法: {config.algorithm}")
        if config.iterative_iterations > 0:
            self._log(f"迭代重构次数: {config.iterative_iterations}")
        self._log(
            f"重构体尺寸: X={config.recon_nx}, Y={config.recon_ny}, Z={config.recon_nz} | "
            f"后处理扩散: {config.refine_iterations} 次, 步长 {config.refine_step:.5f}"
        )

        def on_progress(done: int, total: int, message: str):
            total_safe = max(1, int(total))
            percent = int(
                max(0, min(100, round((float(done) / float(total_safe)) * 100.0)))
            )
            self.progress_bar.setValue(percent)
            if done == 0 or done == total_safe or (done % 5 == 0):
                self._log(message)
            QApplication.processEvents()

        try:
            if self.use_subprocess_checkbox.isChecked():
                result = self._run_reconstruction_subprocess(
                    config=config,
                    projection_files=projection_files,
                    on_progress=on_progress,
                )
            else:
                result = run_fdk_reconstruction(
                    config=config,
                    projection_files=projection_files,
                    progress_callback=on_progress,
                    stop_requested=lambda: self._stop_requested,
                )
            self.progress_bar.setValue(100)
            self._log("-" * 60)
            self._log("阶段3重构完成。")
            self._log(f"切片目录: {result.output_slice_dir}")
            self._log(f"汇总文件: {result.summary_json_path}")
            self._log(f"角度文件: {result.angles_csv_path}")
            self._log(f"计划文件: {result.stage1_plan_json_path}")
            self._log(f"后端: {result.backend_used}")
            if result.backend_note:
                self._log(f"后端说明: {result.backend_note}")
            QMessageBox.information(
                self,
                "完成",
                "阶段3重构完成。\n已输出重构切片与汇总文件。",
            )
        except Exception as e:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "运行失败", str(e))
            self._log(f"运行失败: {e}")
        finally:
            self.run_btn.setEnabled(True)
            self.preview_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.clear_log_btn.setEnabled(True)
            self._stop_requested = False

    def on_preview_middle_clicked(self):
        self.progress_bar.setValue(0)
        config, projection_files = self._build_config_and_files()
        if config is None:
            return

        preview_z = int(max(0, config.recon_nz // 2))
        self._stop_requested = False
        self.run_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.clear_log_btn.setEnabled(False)

        self._log("-" * 60)
        self._log(f"Start preview reconstruction (middle slice): z={preview_z}")

        def on_progress(done: int, total: int, message: str):
            total_safe = max(1, int(total))
            percent = int(
                max(0, min(100, round((float(done) / float(total_safe)) * 100.0)))
            )
            self.progress_bar.setValue(percent)
            if done == 0 or done == total_safe or (done % 5 == 0):
                self._log(message)
            QApplication.processEvents()

        try:
            preview_result = run_reconstruction_preview_slice(
                config=config,
                projection_files=projection_files,
                z_index=preview_z,
                progress_callback=on_progress,
                stop_requested=lambda: self._stop_requested,
            )
            self.progress_bar.setValue(100)
            self._log(f"Preview slice done: z={preview_result.z_index}")
            self._log(f"Preview output: {preview_result.preview_path}")
            if preview_result.backend_note:
                self._log(f"Preview note: {preview_result.backend_note}")
            self._show_preview_dialog(preview_result)
        except Exception as e:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "预览失败", str(e))
            self._log(f"Preview failed: {e}")
        finally:
            self.run_btn.setEnabled(True)
            self.preview_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.clear_log_btn.setEnabled(True)
            self._stop_requested = False

    def _normalize_preview_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"Unsupported preview image shape: {arr.shape}")

        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.uint8)

        valid = arr[finite]
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
            lo = float(np.min(valid))
            hi = float(np.max(valid))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)

        normalized = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        out = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
        out[~finite] = 0
        return out

    def _show_preview_dialog(self, preview_result: ReconstructionPreviewResult):
        image = iio.imread(preview_result.preview_path)
        preview_u8 = self._normalize_preview_image(image)
        h, w = preview_u8.shape
        qimg = QImage(
            preview_u8.data,
            int(w),
            int(h),
            int(preview_u8.strides[0]),
            QImage.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(960, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Preview z={preview_result.z_index}")
        layout = QVBoxLayout(dialog)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        info_label = QLabel(preview_result.preview_path)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.resize(min(1100, pixmap.width() + 80), min(900, pixmap.height() + 150))
        dialog.exec_()

    def on_stop_clicked(self):
        self._stop_requested = True
        if self._worker_process is not None and self._worker_process.poll() is None:
            self._log("已请求停止重构，正在终止阶段3子进程...")
        else:
            self._log("已请求停止重构，将在当前小步骤结束后中止。")

    def _log(self, message: str):
        self.log_text.append(message)
        QApplication.processEvents()

    def _clear_log(self):
        if hasattr(self, "log_text") and self.log_text is not None:
            self.log_text.clear()
