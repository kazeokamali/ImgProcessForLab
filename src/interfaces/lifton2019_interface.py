import json
import os
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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

from src.lifton2019.calibration_builder import build_piecewise_knots
from src.lifton2019.io_loader import (
    collect_image_files,
    load_calibration_set,
    load_paired_subfolder_point_averages,
    load_paired_subfolder_calibration_set,
    load_single_root_calibration_set,
)
from src.lifton2019.models import ProcessingConfig
from src.lifton2019.projection_pipeline import (
    run_lifton2019_model_pipeline,
    run_lifton2019_pipeline,
)
from src.interfaces.ui_theme import apply_interface_theme, set_button_role


class Lifton2019Interface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("lifton2019Interface")
        self.setWindowTitle("Lifton2019 多点平场校正")
        self.resize(1600, 900)
        apply_interface_theme(self)

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
        param_group = self._build_param_group()
        roi_group = self._build_roi_group()
        action_group = self._build_action_group()

        note_label = QLabel(
            "提示：坏点请先在“坏点掩膜”页面处理。\n"
            "本页面流程：dark 校正 -> N点分段线性 flat-field 校正（不做负对数）。\n"
            "支持可变 N 点（N >= 2），并支持“前后标定插值”与“单次标定无漂移”两种模式。\n"
            "如仅需平场标定模型，可点击“仅生成平场标定模型”（不做投影校正）。\n"
            "后续可通过“使用标定模型校正投影”并指定空气 ROI 对样品投影进行修正。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")

        controls_layout.addWidget(folder_group, 0, 0, 1, 2)
        controls_layout.addWidget(param_group, 1, 0)
        controls_layout.addWidget(roi_group, 1, 1)
        controls_layout.addWidget(action_group, 2, 0)
        controls_layout.addWidget(note_label, 2, 1)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setRowStretch(3, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        root_layout.addWidget(controls_scroll, 3)

        self.setCentralWidget(central_widget)

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox("目录设置")
        layout = QVBoxLayout()
        self._drift_mode_widgets = []
        self._single_mode_widgets = []

        self.projection_edit = self._build_folder_row(
            layout, "投影目录", self.on_select_projection_folder
        )
        self.dark_before_edit = self._build_folder_row(
            layout,
            "扫描前 Dark 目录",
            self.on_select_dark_before_folder,
            widget_bucket=self._drift_mode_widgets,
        )
        self.flat_before_edit = self._build_folder_row(
            layout,
            "扫描前 Flat 目录",
            self.on_select_flat_before_folder,
            widget_bucket=self._drift_mode_widgets,
        )
        self.dark_after_edit = self._build_folder_row(
            layout,
            "扫描后 Dark 目录",
            self.on_select_dark_after_folder,
            widget_bucket=self._drift_mode_widgets,
        )
        self.flat_after_edit = self._build_folder_row(
            layout,
            "扫描后 Flat 目录",
            self.on_select_flat_after_folder,
            widget_bucket=self._drift_mode_widgets,
        )
        self.single_calib_root_edit = self._build_folder_row(
            layout,
            "单次标定目录（含子目录）",
            self.on_select_single_calib_root_folder,
            widget_bucket=self._single_mode_widgets,
        )
        self.model_folder_edit = self._build_folder_row(
            layout, "标定模型目录（用于模型校正）", self.on_select_model_folder
        )
        self.output_edit = self._build_folder_row(
            layout, "输出目录", self.on_select_output_folder
        )

        group.setLayout(layout)
        return group

    def _build_param_group(self) -> QGroupBox:
        group = QGroupBox("模型参数")
        layout = QFormLayout()

        self.enable_drift_checkbox = QCheckBox("启用前后标定时间漂移修正")
        self.enable_drift_checkbox.setChecked(True)
        self.enable_drift_checkbox.toggled.connect(self._on_drift_mode_changed)
        layout.addRow(self.enable_drift_checkbox)

        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(2, 64)
        self.num_points_spin.setValue(7)
        layout.addRow("Flat 点数 (N):", self.num_points_spin)

        self.raw_width_spin = QSpinBox()
        self.raw_width_spin.setRange(1, 100000)
        self.raw_width_spin.setValue(2340)
        layout.addRow("RAW 宽度:", self.raw_width_spin)

        self.raw_height_spin = QSpinBox()
        self.raw_height_spin.setRange(1, 100000)
        self.raw_height_spin.setValue(2882)
        layout.addRow("RAW 高度:", self.raw_height_spin)

        self.point_pattern_edit = QLineEdit(r"(\d+(?:\.\d+)?)")
        layout.addRow("点位正则:", self.point_pattern_edit)

        self._on_drift_mode_changed(self.enable_drift_checkbox.isChecked())

        group.setLayout(layout)
        return group

    def _build_roi_group(self) -> QGroupBox:
        group = QGroupBox("参考 ROI（可选）")
        layout = QFormLayout()

        self.use_roi_checkbox = QCheckBox("启用空气 ROI 参考（模型校正必选）")
        self.use_roi_checkbox.toggled.connect(self._on_roi_toggle)
        layout.addRow(self.use_roi_checkbox)

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 100000)
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 100000)
        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 100000)
        self.roi_w_spin.setValue(200)
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 100000)
        self.roi_h_spin.setValue(200)

        layout.addRow("ROI X:", self.roi_x_spin)
        layout.addRow("ROI Y:", self.roi_y_spin)
        layout.addRow("ROI 宽度:", self.roi_w_spin)
        layout.addRow("ROI 高度:", self.roi_h_spin)

        group.setLayout(layout)
        self._on_roi_toggle(False)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("执行操作")
        layout = QVBoxLayout()

        self.validate_btn = QPushButton("校验数据集")
        set_button_role(self.validate_btn, "primary")
        self.validate_btn.clicked.connect(self.on_validate_clicked)
        layout.addWidget(self.validate_btn)

        self.build_calib_btn = QPushButton("仅生成平场标定模型")
        self.build_calib_btn.clicked.connect(self.on_build_calibration_only_clicked)
        layout.addWidget(self.build_calib_btn)

        self.run_btn = QPushButton("开始 N 点校正")
        set_button_role(self.run_btn, "primary")
        self.run_btn.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_btn)

        self.run_model_btn = QPushButton("使用标定模型校正投影")
        set_button_role(self.run_model_btn, "primary")
        self.run_model_btn.clicked.connect(self.on_run_model_correction_clicked)
        layout.addWidget(self.run_model_btn)

        self.clear_log_btn = QPushButton("清空日志")
        set_button_role(self.clear_log_btn, "danger")
        self.clear_log_btn.clicked.connect(self._clear_log)
        layout.addWidget(self.clear_log_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        group.setLayout(layout)
        return group

    def _build_folder_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        callback,
        widget_bucket: Optional[list] = None,
    ) -> QLineEdit:
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
        if widget_bucket is not None:
            widget_bucket.extend([label, edit, button])
        return edit

    def _on_roi_toggle(self, checked: bool):
        self.roi_x_spin.setEnabled(checked)
        self.roi_y_spin.setEnabled(checked)
        self.roi_w_spin.setEnabled(checked)
        self.roi_h_spin.setEnabled(checked)

    def _on_drift_mode_changed(self, enabled: bool):
        # 目录输入始终允许编辑，避免切换模式后无法提前填写路径。
        # 模式开关仅影响运行时“必填项”和处理逻辑。
        for widget in self._drift_mode_widgets:
            widget.setEnabled(True)
        for widget in self._single_mode_widgets:
            widget.setEnabled(True)

    def _select_folder(self, title: str, target_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, title, "")
        if folder:
            target_edit.setText(folder)
            self._log(f"{title}: {folder}")

    def on_select_projection_folder(self):
        self._select_folder("选择投影目录", self.projection_edit)

    def on_select_dark_before_folder(self):
        self._select_folder("选择扫描前 Dark 目录", self.dark_before_edit)

    def on_select_flat_before_folder(self):
        self._select_folder("选择扫描前 Flat 目录", self.flat_before_edit)

    def on_select_dark_after_folder(self):
        self._select_folder("选择扫描后 Dark 目录", self.dark_after_edit)

    def on_select_flat_after_folder(self):
        self._select_folder("选择扫描后 Flat 目录", self.flat_after_edit)

    def on_select_single_calib_root_folder(self):
        self._select_folder("选择单次标定目录（含子目录）", self.single_calib_root_edit)

    def on_select_model_folder(self):
        self._select_folder("选择标定模型目录", self.model_folder_edit)

    def on_select_output_folder(self):
        self._select_folder("选择输出目录", self.output_edit)

    def _build_config(self) -> ProcessingConfig:
        roi: Optional[Tuple[int, int, int, int]] = None
        if self.use_roi_checkbox.isChecked():
            roi = (
                int(self.roi_x_spin.value()),
                int(self.roi_y_spin.value()),
                int(self.roi_w_spin.value()),
                int(self.roi_h_spin.value()),
            )

        return ProcessingConfig(
            num_points=int(self.num_points_spin.value()),
            raw_width=int(self.raw_width_spin.value()),
            raw_height=int(self.raw_height_spin.value()),
            point_pattern=self.point_pattern_edit.text().strip() or r"(\d+(?:\.\d+)?)",
            use_roi_reference=self.use_roi_checkbox.isChecked(),
            reference_roi=roi,
        )

    def _validate_required_paths(self) -> Optional[dict]:
        projection_folder = self.projection_edit.text().strip()
        output_folder = self.output_edit.text().strip()
        if not projection_folder or not output_folder:
            QMessageBox.warning(self, "警告", "请先选择投影目录和输出目录。")
            return None

        if self.enable_drift_checkbox.isChecked():
            dark_before_folder = self.dark_before_edit.text().strip()
            flat_before_folder = self.flat_before_edit.text().strip()
            dark_after_folder = self.dark_after_edit.text().strip()
            flat_after_folder = self.flat_after_edit.text().strip()
            if not all(
                [
                    dark_before_folder,
                    flat_before_folder,
                    dark_after_folder,
                    flat_after_folder,
                ]
            ):
                QMessageBox.warning(self, "警告", "当前模式需要填写扫描前后的 dark/flat 目录。")
                return None
            return {
                "mode": "drift",
                "projection_folder": projection_folder,
                "output_folder": output_folder,
                "dark_before_folder": dark_before_folder,
                "flat_before_folder": flat_before_folder,
                "dark_after_folder": dark_after_folder,
                "flat_after_folder": flat_after_folder,
            }

        single_calib_root = self.single_calib_root_edit.text().strip()
        if not single_calib_root:
            QMessageBox.warning(self, "警告", "当前模式需要填写“单次标定目录（含子目录）”。")
            return None
        return {
            "mode": "single",
            "projection_folder": projection_folder,
            "output_folder": output_folder,
            "single_calib_root": single_calib_root,
        }

    def _validate_required_paths_for_calibration_only(self) -> Optional[dict]:
        output_folder = self.output_edit.text().strip()
        if not output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出目录。")
            return None

        dark_root = self.dark_before_edit.text().strip()
        flat_root = self.flat_before_edit.text().strip()
        if dark_root and flat_root:
            return {
                "mode": "paired_subfolder",
                "dark_root": dark_root,
                "flat_root": flat_root,
                "output_folder": output_folder,
            }

        single_root = self.single_calib_root_edit.text().strip()
        if single_root:
            return {
                "mode": "single_root",
                "single_root": single_root,
                "output_folder": output_folder,
            }

        QMessageBox.warning(
            self,
            "警告",
            "仅生成标定模型时，请填写“扫描前 Dark + 扫描前 Flat”目录，或填写“单次标定目录（含子目录）”。",
        )
        return None

    def _validate_required_paths_for_model_correction(self) -> Optional[dict]:
        projection_folder = self.projection_edit.text().strip()
        output_folder = self.output_edit.text().strip()
        model_folder = self.model_folder_edit.text().strip()
        if not projection_folder or not output_folder or not model_folder:
            QMessageBox.warning(self, "警告", "请先选择投影目录、标定模型目录和输出目录。")
            return None

        if not self.use_roi_checkbox.isChecked():
            QMessageBox.warning(
                self,
                "警告",
                "模型校正需要空气 ROI。请勾选“启用空气 ROI 参考（模型校正必选）”并设置 ROI。",
            )
            return None

        roi = (
            int(self.roi_x_spin.value()),
            int(self.roi_y_spin.value()),
            int(self.roi_w_spin.value()),
            int(self.roi_h_spin.value()),
        )
        return {
            "projection_folder": projection_folder,
            "output_folder": output_folder,
            "model_folder": model_folder,
            "air_roi": roi,
        }

    def on_build_calibration_only_clicked(self):
        maybe_paths = self._validate_required_paths_for_calibration_only()
        if maybe_paths is None:
            return

        config = self._build_config()
        output_folder = maybe_paths["output_folder"]

        self.build_calib_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.run_model_btn.setEnabled(False)

        try:
            flat_knots = None
            dark_knots = None
            if maybe_paths["mode"] == "paired_subfolder":
                point_ids_raw, dark_avgs, flat_avgs, _frame_counts = load_paired_subfolder_point_averages(
                    dark_root_folder=maybe_paths["dark_root"],
                    flat_root_folder=maybe_paths["flat_root"],
                    config=config,
                )
                id_to_dark = {str(pid): dark for pid, dark in zip(point_ids_raw, dark_avgs)}
                id_to_flat = {str(pid): flat for pid, flat in zip(point_ids_raw, flat_avgs)}

                calibration = load_paired_subfolder_calibration_set(
                    dark_root_folder=maybe_paths["dark_root"],
                    flat_root_folder=maybe_paths["flat_root"],
                    config=config,
                )
                self._log("标定模式: Flat/Dark 子目录配对（不做投影校正）")
            else:
                calibration = load_single_root_calibration_set(
                    calibration_root=maybe_paths["single_root"],
                    config=config,
                )
                self._log("标定模式: 单次标定目录（不做投影校正）")
                id_to_flat = {
                    str(pid): arr
                    for pid, arr in zip(calibration.point_ids, calibration.flat_avgs)
                }
                id_to_dark = {
                    str(pid): calibration.dark_avg
                    for pid in calibration.point_ids
                }

            knots = build_piecewise_knots(calibration)
            flat_knots = np.stack(
                [id_to_flat[str(pid)] for pid in knots.point_ids],
                axis=0,
            ).astype(np.float32, copy=False)
            dark_knots = np.stack(
                [id_to_dark[str(pid)] for pid in knots.point_ids],
                axis=0,
            ).astype(np.float32, copy=False)

            os.makedirs(output_folder, exist_ok=True)
            x_path = os.path.join(output_folder, "calibration_x_knots.npy")
            y_path = os.path.join(output_folder, "calibration_y_refs.npy")
            dark_path = os.path.join(output_folder, "calibration_dark_knots.npy")
            flat_path = os.path.join(output_folder, "calibration_flat_knots.npy")
            summary_path = os.path.join(output_folder, "calibration_model_summary.json")

            np.save(x_path, knots.x_knots.astype(np.float32, copy=False))
            np.save(y_path, knots.y_knots.astype(np.float32, copy=False))
            np.save(dark_path, dark_knots)
            np.save(flat_path, flat_knots)

            summary_payload = {
                "num_points": int(knots.x_knots.shape[0]),
                "image_shape": [
                    int(knots.x_knots.shape[1]),
                    int(knots.x_knots.shape[2]),
                ],
                "point_ids": list(knots.point_ids),
                "frame_counts": calibration.frame_counts,
                "x_knots_path": x_path,
                "y_refs_path": y_path,
                "dark_knots_path": dark_path,
                "flat_knots_path": flat_path,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)

            self._log(f"标定模型已生成: {x_path}")
            self._log(f"标定模型已生成: {y_path}")
            self._log(f"标定模型已生成: {dark_path}")
            self._log(f"标定模型已生成: {flat_path}")
            self._log(f"标定摘要已生成: {summary_path}")
            QMessageBox.information(
                self,
                "完成",
                "平场标定模型生成完成（未执行投影图校正）。",
            )
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            self._log(f"生成标定模型失败: {e}")
        finally:
            self.build_calib_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            self.run_model_btn.setEnabled(True)

    def on_run_model_correction_clicked(self):
        maybe_paths = self._validate_required_paths_for_model_correction()
        if maybe_paths is None:
            return

        config = self._build_config()

        self.run_model_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.build_calib_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("-" * 60)
        self._log("开始模型校正。")
        self._log(f"模型目录: {maybe_paths['model_folder']}")
        progress_log_interval = 50

        def on_progress(processed: int, total: int, message: str):
            percent = int((processed / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(percent)
            if (
                processed <= 0
                or processed == 1
                or processed == total
                or (processed % progress_log_interval == 0)
            ):
                self._log(message)
            QApplication.processEvents()

        try:
            result = run_lifton2019_model_pipeline(
                projection_folder=maybe_paths["projection_folder"],
                model_folder=maybe_paths["model_folder"],
                output_folder=maybe_paths["output_folder"],
                config=config,
                air_roi=maybe_paths["air_roi"],
                progress_callback=on_progress,
            )
            self.progress_bar.setValue(100)
            self._log(
                f"模型校正完成。成功: {result.processed_count}, 失败: {result.failed_count}"
            )
            self._log(f"指标 CSV: {result.metrics_csv_path}")
            self._log(f"汇总 JSON: {result.summary_json_path}")
            QMessageBox.information(
                self,
                "完成",
                f"模型校正完成。\n成功: {result.processed_count}\n失败: {result.failed_count}",
            )
        except Exception as e:
            QMessageBox.critical(self, "运行失败", str(e))
            self._log(f"模型校正失败: {e}")
        finally:
            self.run_model_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.build_calib_btn.setEnabled(True)
            self._log("-" * 60)

    def on_validate_clicked(self):
        maybe_paths = self._validate_required_paths()
        if maybe_paths is None:
            return

        projection_folder = maybe_paths["projection_folder"]
        config = self._build_config()

        try:
            projection_files = collect_image_files(projection_folder)
            if len(projection_files) < 1:
                raise ValueError("投影目录中未找到可处理图像文件。")

            if maybe_paths["mode"] == "drift":
                before_set = load_calibration_set(
                    dark_folder=maybe_paths["dark_before_folder"],
                    flat_folder=maybe_paths["flat_before_folder"],
                    config=config,
                )
                after_set = load_calibration_set(
                    dark_folder=maybe_paths["dark_after_folder"],
                    flat_folder=maybe_paths["flat_after_folder"],
                    config=config,
                )
                if before_set.point_ids != after_set.point_ids:
                    raise ValueError(
                        f"点位 ID 不一致：before={before_set.point_ids}, after={after_set.point_ids}"
                    )
                self._log("模式: 前后标定（时间漂移修正）")
            else:
                before_set = load_single_root_calibration_set(
                    calibration_root=maybe_paths["single_calib_root"],
                    config=config,
                )
                after_set = before_set
                self._log("模式: 单次标定（无时间漂移）")

            self._log("校验通过。")
            self._log(f"投影文件数: {len(projection_files)}")
            self._log(
                f"N 点数量: {len(before_set.point_ids)} | 点位 ID: {', '.join(before_set.point_ids)}"
            )
            self._log(
                f"Dark/Flat 尺寸: {before_set.dark_avg.shape[1]}x{before_set.dark_avg.shape[0]}"
            )
        except Exception as e:
            QMessageBox.critical(self, "校验失败", str(e))
            self._log(f"校验失败: {e}")

    def on_run_clicked(self):
        maybe_paths = self._validate_required_paths()
        if maybe_paths is None:
            return

        projection_folder = maybe_paths["projection_folder"]
        output_folder = maybe_paths["output_folder"]
        config = self._build_config()

        self.run_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.build_calib_btn.setEnabled(False)
        self.run_model_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("-" * 60)
        self._log("开始处理。")
        self._log(
            "模式: 前后标定（时间漂移修正）"
            if maybe_paths["mode"] == "drift"
            else "模式: 单次标定（无时间漂移）"
        )
        progress_log_interval = 50

        def on_progress(processed: int, total: int, message: str):
            percent = int((processed / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(percent)
            if (
                processed <= 0
                or processed == 1
                or processed == total
                or (processed % progress_log_interval == 0)
            ):
                self._log(message)
            QApplication.processEvents()

        try:
            calib_before = None
            calib_after = None
            dark_before_folder = maybe_paths.get("dark_before_folder")
            flat_before_folder = maybe_paths.get("flat_before_folder")
            dark_after_folder = maybe_paths.get("dark_after_folder")
            flat_after_folder = maybe_paths.get("flat_after_folder")
            if maybe_paths["mode"] == "single":
                calib_before = load_single_root_calibration_set(
                    calibration_root=maybe_paths["single_calib_root"],
                    config=config,
                )
                calib_after = calib_before

            result = run_lifton2019_pipeline(
                projection_folder=projection_folder,
                dark_before_folder=dark_before_folder,
                flat_before_folder=flat_before_folder,
                dark_after_folder=dark_after_folder,
                flat_after_folder=flat_after_folder,
                output_folder=output_folder,
                config=config,
                calib_before=calib_before,
                calib_after=calib_after,
                progress_callback=on_progress,
            )
            self.progress_bar.setValue(100)
            self._log(
                f"处理完成。成功: {result.processed_count}, 失败: {result.failed_count}"
            )
            self._log(f"指标 CSV: {result.metrics_csv_path}")
            self._log(f"汇总 JSON: {result.summary_json_path}")
            QMessageBox.information(
                self,
                "完成",
                f"处理完成。\n成功: {result.processed_count}\n失败: {result.failed_count}",
            )
        except Exception as e:
            QMessageBox.critical(self, "运行失败", str(e))
            self._log(f"运行失败: {e}")
        finally:
            self.run_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.build_calib_btn.setEnabled(True)
            self.run_model_btn.setEnabled(True)
            self._log("-" * 60)

    def _log(self, message: str):
        self.log_text.append(message)

    def _clear_log(self):
        if hasattr(self, "log_text") and self.log_text is not None:
            self.log_text.clear()
