import json
import os
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
from src.utils.oof_ct_simulation import (
    OOFSimulationConfig,
    compare_recon_to_ground_truth,
    simulate_oof_dataset,
    simulate_oof_dataset_from_slice_folders,
)


class OOFCTSimulationInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("oofCTSimulationInterface")
        self.setWindowTitle("超视野CT误差模拟与分析")
        self.resize(1600, 900)
        apply_interface_theme(self)
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(220)

        self.source_mode_group = self._build_source_mode_group()
        self.folder_group = self._build_folder_group()
        self.sim_group = self._build_sim_group()
        self.oof_group = self._build_oof_group()

        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        root_layout.addWidget(log_group, 2)

        controls_widget = QWidget()
        controls_layout = QGridLayout(controls_widget)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(12)

        compare_group = self._build_compare_group()
        action_group = self._build_action_group()

        note_label = QLabel(
            "用途：模拟 local/interior tomography 场景下的超视野截断误差。\n"
            "流程：选择数据来源 -> 生成 base/OOF 投影 -> 用重构页面各自重构 -> 回到本页做相似度对比。\n"
            "说明：探测器列数固定时，local 采用“方图幅 + 圆形有效区(直径=探测器列数)”语义；"
            "full ground truth 包含 FOV 外材料（用于观察超视野区域）。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")

        controls_layout.addWidget(self.source_mode_group, 0, 0, 1, 2)
        controls_layout.addWidget(self.folder_group, 1, 0, 1, 2)
        controls_layout.addWidget(self.sim_group, 2, 0)
        controls_layout.addWidget(self.oof_group, 2, 1)
        controls_layout.addWidget(compare_group, 3, 0)
        controls_layout.addWidget(action_group, 3, 1)
        controls_layout.addWidget(note_label, 4, 0, 1, 2)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setRowStretch(5, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        root_layout.addWidget(controls_scroll, 3)

        self.setCentralWidget(central_widget)
        self._on_source_mode_changed()

    def _build_source_mode_group(self) -> QGroupBox:
        group = QGroupBox("仿真数据来源")
        layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("来源模式:"))
        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItem("合成体模（内置）", "synthetic")
        self.source_mode_combo.addItem("用户原始切片（自定义）", "user_slices")
        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        mode_row.addWidget(self.source_mode_combo, 1)
        layout.addLayout(mode_row)

        user_base_row = QHBoxLayout()
        user_base_row.addWidget(QLabel("用户Base full切片目录"))
        self.user_base_full_edit = QLineEdit()
        self.user_base_full_edit.setReadOnly(True)
        self.user_base_full_btn = QPushButton("选择")
        self.user_base_full_btn.clicked.connect(self.on_select_user_base_full_folder)
        user_base_row.addWidget(self.user_base_full_edit, 1)
        user_base_row.addWidget(self.user_base_full_btn)
        layout.addLayout(user_base_row)

        user_oof_row = QHBoxLayout()
        user_oof_row.addWidget(QLabel("用户OOF full切片目录"))
        self.user_oof_full_edit = QLineEdit()
        self.user_oof_full_edit.setReadOnly(True)
        self.user_oof_full_btn = QPushButton("选择")
        self.user_oof_full_btn.clicked.connect(self.on_select_user_oof_full_folder)
        user_oof_row.addWidget(self.user_oof_full_edit, 1)
        user_oof_row.addWidget(self.user_oof_full_btn)
        layout.addLayout(user_oof_row)

        tip_label = QLabel(
            "提示：用户模式下，探测器列数即核心圆直径；Base/OOF full切片将直接作为仿真输入。"
        )
        tip_label.setWordWrap(True)
        tip_label.setStyleSheet("color: #555;")
        layout.addWidget(tip_label)

        group.setLayout(layout)
        return group

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox("目录设置")
        layout = QVBoxLayout()
        self.output_edit = self._build_folder_row(
            layout, "仿真输出目录", self.on_select_output_folder
        )
        self.gt_edit = self._build_folder_row(
            layout, "Ground Truth切片目录", self.on_select_gt_folder
        )
        self.recon_base_edit = self._build_folder_row(
            layout, "Base重构切片目录", self.on_select_recon_base_folder
        )
        self.recon_oof_edit = self._build_folder_row(
            layout, "OOF重构切片目录", self.on_select_recon_oof_folder
        )
        group.setLayout(layout)
        return group

    def _build_sim_group(self) -> QGroupBox:
        group = QGroupBox("仿真参数")
        layout = QFormLayout()

        self.sim_z_spin = QSpinBox()
        self.sim_z_spin.setRange(8, 4096)
        self.sim_z_spin.setValue(256)
        layout.addRow("仿真体层数 Z:", self.sim_z_spin)

        self.detector_width_spin = QSpinBox()
        self.detector_width_spin.setRange(16, 4096)
        self.detector_width_spin.setValue(100)
        layout.addRow("投影线条数/核心圆直径(像素):", self.detector_width_spin)

        self.num_angles_spin = QSpinBox()
        self.num_angles_spin.setRange(16, 2000)
        self.num_angles_spin.setValue(360)
        layout.addRow("投影角度数:", self.num_angles_spin)

        self.base_radius_px_spin = QDoubleSpinBox()
        self.base_radius_px_spin.setRange(2.0, 10000.0)
        self.base_radius_px_spin.setSingleStep(1.0)
        self.base_radius_px_spin.setValue(40.0)
        self.base_radius_px_spin.setDecimals(2)
        layout.addRow("基础圆柱半径(像素):", self.base_radius_px_spin)

        self.base_intensity_spin = QDoubleSpinBox()
        self.base_intensity_spin.setRange(0.01, 100.0)
        self.base_intensity_spin.setValue(1.0)
        layout.addRow("基础材料衰减系数:", self.base_intensity_spin)

        group.setLayout(layout)
        return group

    def _build_oof_group(self) -> QGroupBox:
        group = QGroupBox("超视野材料参数")
        layout = QFormLayout()

        self.add_oof_cb = QCheckBox("启用超视野材料")
        self.add_oof_cb.setChecked(True)
        layout.addRow(self.add_oof_cb)

        self.oof_type_combo = QComboBox()
        self.oof_type_combo.addItems(["环形壳层", "双侧团簇"])
        layout.addRow("超视野材料类型:", self.oof_type_combo)

        self.oof_intensity_spin = QDoubleSpinBox()
        self.oof_intensity_spin.setRange(0.001, 100.0)
        self.oof_intensity_spin.setValue(0.4)
        layout.addRow("超视野材料衰减系数:", self.oof_intensity_spin)

        self.oof_size_px_spin = QDoubleSpinBox()
        self.oof_size_px_spin.setRange(2.0, 10000.0)
        self.oof_size_px_spin.setSingleStep(1.0)
        self.oof_size_px_spin.setValue(20.0)
        self.oof_size_px_spin.setDecimals(2)
        layout.addRow("超视野材料尺寸(像素):", self.oof_size_px_spin)

        group.setLayout(layout)
        return group

    def _build_compare_group(self) -> QGroupBox:
        group = QGroupBox("相似度分析参数")
        layout = QFormLayout()

        self.enable_roi_cb = QCheckBox("启用ROI对比")
        self.enable_roi_cb.setChecked(False)
        self.enable_roi_cb.stateChanged.connect(self._on_roi_toggle)
        layout.addRow(self.enable_roi_cb)

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 100000)
        self.roi_x_spin.setValue(100)
        layout.addRow("ROI x:", self.roi_x_spin)

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 100000)
        self.roi_y_spin.setValue(100)
        layout.addRow("ROI y:", self.roi_y_spin)

        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 100000)
        self.roi_w_spin.setValue(128)
        layout.addRow("ROI width:", self.roi_w_spin)

        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 100000)
        self.roi_h_spin.setValue(128)
        layout.addRow("ROI height:", self.roi_h_spin)

        self.slice_step_spin = QSpinBox()
        self.slice_step_spin.setRange(1, 1000)
        self.slice_step_spin.setValue(1)
        layout.addRow("切片采样步长:", self.slice_step_spin)

        group.setLayout(layout)
        self._on_roi_toggle()
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("执行操作")
        layout = QVBoxLayout()

        self.generate_btn = QPushButton("1) 生成仿真数据（Base + OOF）")
        set_button_role(self.generate_btn, "primary")
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        layout.addWidget(self.generate_btn)

        self.compare_btn = QPushButton("2) 计算重构相似度报告")
        set_button_role(self.compare_btn, "primary")
        self.compare_btn.clicked.connect(self.on_compare_clicked)
        layout.addWidget(self.compare_btn)

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

    def _build_folder_row(
        self, parent_layout: QVBoxLayout, label_text: str, callback
    ) -> QLineEdit:
        row = QHBoxLayout()
        label = QLabel(label_text)
        edit = QLineEdit()
        edit.setReadOnly(True)
        button = QPushButton("选择")
        button.clicked.connect(callback)
        row.addWidget(label)
        row.addWidget(edit, 1)
        row.addWidget(button)
        parent_layout.addLayout(row)
        return edit

    def _on_roi_toggle(self):
        enabled = self.enable_roi_cb.isChecked()
        self.roi_x_spin.setEnabled(enabled)
        self.roi_y_spin.setEnabled(enabled)
        self.roi_w_spin.setEnabled(enabled)
        self.roi_h_spin.setEnabled(enabled)

    def _on_source_mode_changed(self):
        mode = self.source_mode_combo.currentData()
        is_user_mode = mode == "user_slices"

        self.user_base_full_edit.setEnabled(is_user_mode)
        self.user_oof_full_edit.setEnabled(is_user_mode)
        self.user_base_full_btn.setEnabled(is_user_mode)
        self.user_oof_full_btn.setEnabled(is_user_mode)

        if hasattr(self, "sim_z_spin"):
            self.sim_z_spin.setEnabled(not is_user_mode)
        if hasattr(self, "base_radius_px_spin"):
            self.base_radius_px_spin.setEnabled(not is_user_mode)
        if hasattr(self, "base_intensity_spin"):
            self.base_intensity_spin.setEnabled(not is_user_mode)
        if hasattr(self, "oof_group"):
            self.oof_group.setEnabled(not is_user_mode)

        if hasattr(self, "log_text"):
            if is_user_mode:
                self._log("已切换到【用户原始切片】模式。")
            else:
                self._log("已切换到【合成体模】模式。")

    def on_select_output_folder(self):
        self._select_folder("选择仿真输出目录", self.output_edit)

    def on_select_gt_folder(self):
        self._select_folder("选择 Ground Truth 切片目录", self.gt_edit)

    def on_select_recon_base_folder(self):
        self._select_folder("选择 Base 重构切片目录", self.recon_base_edit)

    def on_select_recon_oof_folder(self):
        self._select_folder("选择 OOF 重构切片目录", self.recon_oof_edit)

    def on_select_user_base_full_folder(self):
        self._select_folder("选择 用户Base full切片目录", self.user_base_full_edit)

    def on_select_user_oof_full_folder(self):
        self._select_folder("选择 用户OOF full切片目录", self.user_oof_full_edit)

    def _select_folder(self, title: str, target_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, title, "")
        if folder:
            target_edit.setText(folder)
            self._log(f"{title}: {folder}")

    def _log(self, message: str):
        self.log_text.append(message)
        QApplication.processEvents()

    def _build_sim_config(self) -> OOFSimulationConfig:
        oof_map = {"环形壳层": "ring_shell", "双侧团簇": "side_blobs"}
        return OOFSimulationConfig(
            sim_z=int(self.sim_z_spin.value()),
            detector_width=int(self.detector_width_spin.value()),
            num_angles=int(self.num_angles_spin.value()),
            phantom_type="cylinder",
            base_radius_px=float(self.base_radius_px_spin.value()),
            base_intensity=float(self.base_intensity_spin.value()),
            add_oof_material=bool(self.add_oof_cb.isChecked()),
            oof_type=oof_map.get(self.oof_type_combo.currentText(), "ring_shell"),
            oof_intensity=float(self.oof_intensity_spin.value()),
            oof_size_px=float(self.oof_size_px_spin.value()),
        )

    def on_generate_clicked(self):
        output_root = self.output_edit.text().strip()
        if not output_root:
            QMessageBox.warning(self, "提示", "请先选择仿真输出目录。")
            return

        self.progress_bar.setValue(0)
        self.generate_btn.setEnabled(False)
        self.compare_btn.setEnabled(False)
        try:
            mode = self.source_mode_combo.currentData()
            self._log("-" * 60)
            self._log("开始生成超视野CT仿真数据...")

            def on_progress(done: int, total: int, msg: str):
                pct = int(
                    max(
                        0, min(100, round((float(done) / max(1, float(total))) * 100.0))
                    )
                )
                self.progress_bar.setValue(pct)
                if done == 0 or done == total or done % 10 == 0:
                    self._log(msg)
                QApplication.processEvents()

            if mode == "user_slices":
                base_full_dir = self.user_base_full_edit.text().strip()
                oof_full_dir = self.user_oof_full_edit.text().strip()
                if not base_full_dir:
                    QMessageBox.warning(
                        self, "提示", "请先选择 用户Base full切片目录。"
                    )
                    return

                self._log(
                    f"用户模式参数: detector_width={int(self.detector_width_spin.value())}, "
                    f"num_angles={int(self.num_angles_spin.value())}"
                )
                self._log(f"用户Base full目录: {base_full_dir}")
                self._log(
                    f"用户OOF full目录: {oof_full_dir or '(未提供，默认使用Base)'}"
                )

                result = simulate_oof_dataset_from_slice_folders(
                    base_full_slice_dir=base_full_dir,
                    oof_full_slice_dir=(oof_full_dir or None),
                    detector_width=int(self.detector_width_spin.value()),
                    num_angles=int(self.num_angles_spin.value()),
                    output_root=output_root,
                    progress_callback=on_progress,
                )
            else:
                cfg = self._build_sim_config()
                self._log(
                    f"合成模式参数: sim_z={cfg.sim_z}, detector_width={cfg.detector_width}, "
                    f"num_angles={cfg.num_angles}, base_radius_px={cfg.base_radius_px}, "
                    f"base_mu={cfg.base_intensity}, add_oof={cfg.add_oof_material}"
                )
                result = simulate_oof_dataset(
                    config=cfg,
                    output_root=output_root,
                    progress_callback=on_progress,
                )

            self.progress_bar.setValue(100)
            self._log("仿真数据生成完成。")
            self._log(f"自动仿真横向尺寸 XY: {result['sim_xy_auto']}")
            self._log(f"Base 投影目录: {result['base_projection_dir']}")
            self._log(f"OOF 投影目录: {result['oof_projection_dir']}")
            self._log(
                f"Base Ground Truth(局部/FOV): {result['base_ground_truth_local_dir']}"
            )
            self._log(
                f"Base Ground Truth(完整/含FOV外): {result['base_ground_truth_full_dir']}"
            )
            self._log(
                f"OOF Ground Truth(局部/FOV): {result['oof_ground_truth_local_dir']}"
            )
            self._log(
                f"OOF Ground Truth(完整/含FOV外): {result['oof_ground_truth_full_dir']}"
            )
            self._log(f"元数据: {result['metadata_path']}")

            # 默认分析时使用局部 ground truth（与截断投影对应）
            self.gt_edit.setText(result["base_ground_truth_local_dir"])
            QMessageBox.information(
                self,
                "完成",
                "仿真数据已生成。\n"
                "建议重构时使用 Base/OOF 投影目录。\n"
                "相似度分析默认使用 local ground truth；若要直观看到超视野材料，请查看 OOF 的 full ground truth 目录。",
            )
        except Exception as e:
            self.progress_bar.setValue(0)
            self._log(f"仿真失败: {e}")
            QMessageBox.critical(self, "失败", str(e))
        finally:
            self.generate_btn.setEnabled(True)
            self.compare_btn.setEnabled(True)

    def _get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.enable_roi_cb.isChecked():
            return None
        return (
            int(self.roi_x_spin.value()),
            int(self.roi_y_spin.value()),
            int(self.roi_w_spin.value()),
            int(self.roi_h_spin.value()),
        )

    def on_compare_clicked(self):
        gt_dir = self.gt_edit.text().strip()
        recon_base_dir = self.recon_base_edit.text().strip()
        recon_oof_dir = self.recon_oof_edit.text().strip()
        if not gt_dir:
            QMessageBox.warning(self, "提示", "请先选择 Ground Truth 切片目录。")
            return
        if not recon_base_dir and not recon_oof_dir:
            QMessageBox.warning(
                self, "提示", "请至少选择一个重构切片目录（Base 或 OOF）。"
            )
            return

        roi = self._get_roi()
        step = int(self.slice_step_spin.value())
        self.progress_bar.setValue(0)
        self.compare_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        try:
            self._log("-" * 60)
            self._log("开始计算相似度指标...")
            report = {
                "ground_truth_dir": gt_dir,
                "roi": list(roi) if roi is not None else None,
                "slice_step": step,
            }

            if recon_base_dir:
                self._log("计算 Base 重构指标...")
                base_metrics = compare_recon_to_ground_truth(
                    ground_truth_dir=gt_dir,
                    recon_dir=recon_base_dir,
                    roi=roi,
                    slice_step=step,
                )
                report["base_metrics"] = base_metrics
                self._log(
                    f"Base: RMSE={base_metrics['rmse_mean']:.6f}, NCC={base_metrics['ncc_mean']:.6f}"
                )

            self.progress_bar.setValue(50)

            if recon_oof_dir:
                self._log("计算 OOF 重构指标...")
                oof_metrics = compare_recon_to_ground_truth(
                    ground_truth_dir=gt_dir,
                    recon_dir=recon_oof_dir,
                    roi=roi,
                    slice_step=step,
                )
                report["oof_metrics"] = oof_metrics
                self._log(
                    f"OOF: RMSE={oof_metrics['rmse_mean']:.6f}, NCC={oof_metrics['ncc_mean']:.6f}"
                )

            if ("base_metrics" in report) and ("oof_metrics" in report):
                report["delta_oof_minus_base"] = {
                    "rmse_mean": report["oof_metrics"]["rmse_mean"]
                    - report["base_metrics"]["rmse_mean"],
                    "mae_mean": report["oof_metrics"]["mae_mean"]
                    - report["base_metrics"]["mae_mean"],
                    "psnr_mean": report["oof_metrics"]["psnr_mean"]
                    - report["base_metrics"]["psnr_mean"],
                    "ncc_mean": report["oof_metrics"]["ncc_mean"]
                    - report["base_metrics"]["ncc_mean"],
                }
                delta = report["delta_oof_minus_base"]
                self._log(
                    f"OOF 相对 Base 变化: ΔRMSE={delta['rmse_mean']:+.6f}, "
                    f"ΔPSNR={delta['psnr_mean']:+.6f}, ΔNCC={delta['ncc_mean']:+.6f}"
                )

            out_root = self.output_edit.text().strip() or os.path.dirname(gt_dir)
            os.makedirs(out_root, exist_ok=True)
            report_path = os.path.join(out_root, "oof_similarity_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self.progress_bar.setValue(100)
            self._log(f"相似度报告已保存: {report_path}")
            QMessageBox.information(self, "完成", f"相似度报告已生成：\n{report_path}")
        except Exception as e:
            self.progress_bar.setValue(0)
            self._log(f"相似度计算失败: {e}")
            QMessageBox.critical(self, "失败", str(e))
        finally:
            self.compare_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
