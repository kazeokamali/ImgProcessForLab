import os
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.lifton2019.io_loader import collect_image_files, load_calibration_set
from src.lifton2019.models import ProcessingConfig
from src.lifton2019.projection_pipeline import run_lifton2019_pipeline


class Lifton2019Interface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("lifton2019Interface")
        self.setWindowTitle("Lifton2019 多点平场校正")
        self.resize(1600, 900)

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        root_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        left_layout.addWidget(self._build_folder_group())
        left_layout.addWidget(self._build_param_group())
        left_layout.addWidget(self._build_roi_group())
        left_layout.addWidget(self._build_action_group())

        note_label = QLabel(
            "提示：坏点请先在“坏点掩膜”页面处理。\n"
            "本页面流程：dark 校正 -> N点分段线性 flat-field 校正 -> 负对数。\n"
            "支持可变 N 点（N >= 2），并支持前后标定时间插值。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")
        left_layout.addWidget(note_label)
        left_layout.addStretch(1)

        right_layout.addWidget(QLabel("处理日志"))
        right_layout.addWidget(self.log_text)

        root_layout.addLayout(left_layout, 2)
        root_layout.addLayout(right_layout, 3)
        self.setCentralWidget(central_widget)

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox("目录设置")
        layout = QVBoxLayout()

        self.projection_edit = self._build_folder_row(
            layout, "投影目录", self.on_select_projection_folder
        )
        self.dark_before_edit = self._build_folder_row(
            layout, "扫描前 Dark 目录", self.on_select_dark_before_folder
        )
        self.flat_before_edit = self._build_folder_row(
            layout, "扫描前 Flat 目录", self.on_select_flat_before_folder
        )
        self.dark_after_edit = self._build_folder_row(
            layout, "扫描后 Dark 目录", self.on_select_dark_after_folder
        )
        self.flat_after_edit = self._build_folder_row(
            layout, "扫描后 Flat 目录", self.on_select_flat_after_folder
        )
        self.output_edit = self._build_folder_row(
            layout, "输出目录", self.on_select_output_folder
        )

        group.setLayout(layout)
        return group

    def _build_param_group(self) -> QGroupBox:
        group = QGroupBox("模型参数")
        layout = QFormLayout()

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

        self.point_pattern_edit = QLineEdit(r"(\d+)")
        layout.addRow("点位正则:", self.point_pattern_edit)

        group.setLayout(layout)
        return group

    def _build_roi_group(self) -> QGroupBox:
        group = QGroupBox("参考 ROI（可选）")
        layout = QFormLayout()

        self.use_roi_checkbox = QCheckBox("使用 ROI 均值作为参考强度")
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
        self.validate_btn.clicked.connect(self.on_validate_clicked)
        layout.addWidget(self.validate_btn)

        self.run_btn = QPushButton("开始 N 点校正")
        self.run_btn.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_btn)

        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self._clear_log)
        layout.addWidget(self.clear_log_btn)

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
        row.addWidget(edit)
        row.addWidget(button)
        parent_layout.addLayout(row)
        return edit

    def _on_roi_toggle(self, checked: bool):
        self.roi_x_spin.setEnabled(checked)
        self.roi_y_spin.setEnabled(checked)
        self.roi_w_spin.setEnabled(checked)
        self.roi_h_spin.setEnabled(checked)

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
            point_pattern=self.point_pattern_edit.text().strip() or r"(\d+)",
            use_roi_reference=self.use_roi_checkbox.isChecked(),
            reference_roi=roi,
        )

    def _validate_required_paths(self) -> Optional[Tuple[str, str, str, str, str, str]]:
        paths = (
            self.projection_edit.text().strip(),
            self.dark_before_edit.text().strip(),
            self.flat_before_edit.text().strip(),
            self.dark_after_edit.text().strip(),
            self.flat_after_edit.text().strip(),
            self.output_edit.text().strip(),
        )
        if not all(paths):
            QMessageBox.warning(self, "警告", "请先选择所有必需目录。")
            return None
        return paths

    def on_validate_clicked(self):
        maybe_paths = self._validate_required_paths()
        if maybe_paths is None:
            return

        (
            projection_folder,
            dark_before_folder,
            flat_before_folder,
            dark_after_folder,
            flat_after_folder,
            _output_folder,
        ) = maybe_paths
        config = self._build_config()

        try:
            projection_files = collect_image_files(projection_folder)
            if len(projection_files) < 1:
                raise ValueError("投影目录中未找到可处理图像文件。")

            before_set = load_calibration_set(
                dark_folder=dark_before_folder,
                flat_folder=flat_before_folder,
                config=config,
            )
            after_set = load_calibration_set(
                dark_folder=dark_after_folder,
                flat_folder=flat_after_folder,
                config=config,
            )

            if before_set.point_ids != after_set.point_ids:
                raise ValueError(
                    f"点位 ID 不一致：before={before_set.point_ids}, after={after_set.point_ids}"
                )

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

        (
            projection_folder,
            dark_before_folder,
            flat_before_folder,
            dark_after_folder,
            flat_after_folder,
            output_folder,
        ) = maybe_paths
        config = self._build_config()

        self.run_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("-" * 60)
        self._log("开始处理。")

        def on_progress(processed: int, total: int, message: str):
            percent = int((processed / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(percent)
            self._log(message)
            QApplication.processEvents()

        try:
            result = run_lifton2019_pipeline(
                projection_folder=projection_folder,
                dark_before_folder=dark_before_folder,
                flat_before_folder=flat_before_folder,
                dark_after_folder=dark_after_folder,
                flat_after_folder=flat_after_folder,
                output_folder=output_folder,
                config=config,
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
            self._log("-" * 60)

    def _log(self, message: str):
        self.log_text.append(message)

    def _clear_log(self):
        if hasattr(self, "log_text") and self.log_text is not None:
            self.log_text.clear()
