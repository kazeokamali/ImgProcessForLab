import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from src.interfaces.ui_theme import apply_interface_theme, set_button_role


class SpeedPlotDialog(QDialog):
    def __init__(self, plot_payload: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(plot_payload.get("title", "速度曲线"))
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self._render_plot(plot_payload)

    def _render_plot(self, plot_payload: dict):
        ax = self.figure.add_subplot(111)
        x_values = plot_payload.get("x_values", [])
        series = plot_payload.get("series", [])

        for label, y_values in series:
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.2,
                markersize=2.5,
                label=label,
            )

        ax.set_xlabel(plot_payload.get("x_label", "frame-index"))
        ax.set_ylabel(plot_payload.get("y_label", "velocity"))
        ax.set_title(plot_payload.get("title", "v-curve"))
        ax.grid(True, alpha=0.3)

        if series and len(series) <= 20:
            ax.legend(loc="best", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()


class WaveSpeedInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("waveSpeedInterface")
        self.setWindowTitle("波阵面传播速度分析")
        self.resize(1400, 900)
        apply_interface_theme(self)

        self.input_folder: Optional[str] = None
        self.output_folder: Optional[str] = None
        self.last_plot_payload: Optional[dict] = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        folder_group = QGroupBox("数据目录")
        folder_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.input_line_edit = QLineEdit()
        self.input_line_edit.setReadOnly(True)
        input_button = QPushButton("选择输入文件夹(CSV)")
        input_button.clicked.connect(self.on_select_input_folder)
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(input_button)
        folder_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.output_line_edit = QLineEdit()
        self.output_line_edit.setReadOnly(True)
        output_button = QPushButton("选择输出文件夹")
        output_button.clicked.connect(self.on_select_output_folder)
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(output_button)
        folder_layout.addLayout(output_layout)

        folder_group.setLayout(folder_layout)
        left_panel.addWidget(folder_group)

        range_group = QGroupBox("行范围")
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("起始行:"))
        self.row_start_spin = QSpinBox()
        self.row_start_spin.setRange(1, 1000000)
        self.row_start_spin.setValue(1)
        range_layout.addWidget(self.row_start_spin)

        range_layout.addWidget(QLabel("结束行:"))
        self.row_end_spin = QSpinBox()
        self.row_end_spin.setRange(1, 1000000)
        self.row_end_spin.setValue(100)
        range_layout.addWidget(self.row_end_spin)
        range_group.setLayout(range_layout)
        left_panel.addWidget(range_group)

        param_group = QGroupBox("速度换算参数")
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("空间标定(单位/像素):"))
        self.pixel_scale_spin = QDoubleSpinBox()
        self.pixel_scale_spin.setRange(1e-9, 1e9)
        self.pixel_scale_spin.setDecimals(9)
        self.pixel_scale_spin.setValue(1.0)
        self.pixel_scale_spin.setSingleStep(0.01)
        param_layout.addWidget(self.pixel_scale_spin)

        param_layout.addWidget(QLabel("帧间隔(秒):"))
        self.frame_interval_spin = QDoubleSpinBox()
        self.frame_interval_spin.setRange(1e-9, 1e9)
        self.frame_interval_spin.setDecimals(9)
        self.frame_interval_spin.setValue(1.0)
        self.frame_interval_spin.setSingleStep(0.001)
        param_layout.addWidget(self.frame_interval_spin)
        param_group.setLayout(param_layout)
        left_panel.addWidget(param_group)

        action_group = QGroupBox("算法按钮")
        action_layout = QVBoxLayout()

        per_row_button = QPushButton("1. 计算按行传播速度")
        set_button_role(per_row_button, "primary")
        per_row_button.clicked.connect(self.on_compute_per_row_speed)
        action_layout.addWidget(per_row_button)

        leftmost_button = QPushButton("2. 计算最左传播点速度")
        set_button_role(leftmost_button, "primary")
        leftmost_button.clicked.connect(self.on_compute_leftmost_speed)
        action_layout.addWidget(leftmost_button)

        average_button = QPushButton("3. 计算区间平均传播速度")
        set_button_role(average_button, "primary")
        average_button.clicked.connect(self.on_compute_average_speed)
        action_layout.addWidget(average_button)

        plot_button = QPushButton("绘制速度曲线(弹窗)")
        plot_button.clicked.connect(self.on_plot_speed_curves)
        action_layout.addWidget(plot_button)

        clear_button = QPushButton("清空日志")
        set_button_role(clear_button, "danger")
        clear_button.clicked.connect(lambda: self.log_text.clear())
        action_layout.addWidget(clear_button)

        action_group.setLayout(action_layout)
        left_panel.addWidget(action_group)

        note_label = QLabel(
            "说明:\n"
            "1. 输入目录中放置“突变点提取”导出的 CSV 文件。\n"
            "2. 速度定义为“向左传播速度”= (前一帧Y - 后一帧Y) * 空间标定 / 帧间隔。\n"
            "3. 输出目录会生成三类算法对应的 CSV 结果。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #555;")
        left_panel.addWidget(note_label)
        left_panel.addStretch(1)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_panel.addWidget(QLabel("处理日志"))
        right_panel.addWidget(self.log_text)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        self.setCentralWidget(central_widget)

    def on_select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择CSV输入文件夹", "")
        if folder:
            self.input_folder = folder
            self.input_line_edit.setText(folder)
            self._log(f"输入目录: {folder}")

    def on_select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择速度数据输出文件夹", "")
        if folder:
            self.output_folder = folder
            self.output_line_edit.setText(folder)
            self._log(f"输出目录: {folder}")

    def on_compute_per_row_speed(self):
        prepared = self._prepare_common_inputs()
        if prepared is None:
            return

        csv_files = prepared["csv_files"]
        rows = prepared["rows"]
        matrix = prepared["matrix"]
        frame_interval = prepared["frame_interval"]
        pixel_scale = prepared["pixel_scale"]
        output_folder = prepared["output_folder"]

        frame_count = len(csv_files)
        pair_count = frame_count - 1

        speed_matrix = np.full((pair_count, len(rows)), np.nan, dtype=float)
        long_rows = []

        for row_idx, row_value in enumerate(rows):
            y_values = matrix[:, row_idx]
            delta_y, speeds = self._compute_pair_speed(
                y_values, pixel_scale, frame_interval
            )
            speed_matrix[:, row_idx] = speeds

            for pair_idx in range(pair_count):
                long_rows.append(
                    [
                        pair_idx + 1,
                        os.path.basename(csv_files[pair_idx]),
                        os.path.basename(csv_files[pair_idx + 1]),
                        row_value,
                        self._to_csv_value(y_values[pair_idx]),
                        self._to_csv_value(y_values[pair_idx + 1]),
                        self._to_csv_value(delta_y[pair_idx]),
                        self._to_csv_value(speeds[pair_idx]),
                    ]
                )

        summary_rows = []
        for row_idx, row_value in enumerate(rows):
            row_speed = speed_matrix[:, row_idx]
            valid_speed = row_speed[np.isfinite(row_speed)]
            summary_rows.append(
                [
                    row_value,
                    int(valid_speed.size),
                    (
                        self._to_csv_value(np.mean(valid_speed))
                        if valid_speed.size
                        else "nan"
                    ),
                    (
                        self._to_csv_value(np.std(valid_speed))
                        if valid_speed.size
                        else "nan"
                    ),
                    (
                        self._to_csv_value(np.min(valid_speed))
                        if valid_speed.size
                        else "nan"
                    ),
                    (
                        self._to_csv_value(np.max(valid_speed))
                        if valid_speed.size
                        else "nan"
                    ),
                ]
            )

        wide_header = ["frame_pair_index", "start_file", "end_file"] + [
            f"row_{r}" for r in rows
        ]
        wide_rows = []
        for pair_idx in range(pair_count):
            row_data = [
                pair_idx + 1,
                os.path.basename(csv_files[pair_idx]),
                os.path.basename(csv_files[pair_idx + 1]),
            ]
            for row_idx in range(len(rows)):
                row_data.append(self._to_csv_value(speed_matrix[pair_idx, row_idx]))
            wide_rows.append(row_data)

        long_path = os.path.join(output_folder, "speed_per_row_long.csv")
        wide_path = os.path.join(output_folder, "speed_per_row_wide.csv")
        summary_path = os.path.join(output_folder, "speed_per_row_summary.csv")
        self._save_csv(
            long_path,
            [
                "frame_pair_index",
                "start_file",
                "end_file",
                "row",
                "y_start",
                "y_end",
                "delta_y",
                "leftward_speed",
            ],
            long_rows,
        )
        self._save_csv(wide_path, wide_header, wide_rows)
        self._save_csv(
            summary_path,
            ["row", "valid_pairs", "mean_speed", "std_speed", "min_speed", "max_speed"],
            summary_rows,
        )

        self.last_plot_payload = {
            "title": "按行传播速度曲线",
            "x_label": "帧对索引",
            "y_label": "向左传播速度",
            "x_values": list(range(1, pair_count + 1)),
            "series": [
                (f"row_{rows[i]}", speed_matrix[:, i].tolist())
                for i in range(len(rows))
            ],
        }

        self._log(f"已输出: {long_path}")
        self._log(f"已输出: {wide_path}")
        self._log(f"已输出: {summary_path}")
        self._log("算法1完成: 已生成按行速度曲线数据。")

    def on_compute_leftmost_speed(self):
        prepared = self._prepare_common_inputs()
        if prepared is None:
            return

        csv_files = prepared["csv_files"]
        rows = prepared["rows"]
        matrix = prepared["matrix"]
        frame_interval = prepared["frame_interval"]
        pixel_scale = prepared["pixel_scale"]
        output_folder = prepared["output_folder"]

        frame_count = len(csv_files)
        leftmost_y = np.full(frame_count, np.nan, dtype=float)
        leftmost_row = np.full(frame_count, np.nan, dtype=float)

        for frame_idx in range(frame_count):
            frame_values = matrix[frame_idx, :]
            valid_idx = np.where(np.isfinite(frame_values))[0]
            if valid_idx.size == 0:
                continue
            local_min_pos = valid_idx[np.argmin(frame_values[valid_idx])]
            leftmost_y[frame_idx] = frame_values[local_min_pos]
            leftmost_row[frame_idx] = rows[local_min_pos]

        delta_y, speeds = self._compute_pair_speed(
            leftmost_y, pixel_scale, frame_interval
        )

        position_rows = []
        for frame_idx, csv_path in enumerate(csv_files):
            position_rows.append(
                [
                    frame_idx + 1,
                    os.path.basename(csv_path),
                    self._to_csv_value(leftmost_row[frame_idx]),
                    self._to_csv_value(leftmost_y[frame_idx]),
                ]
            )

        speed_rows = []
        for pair_idx in range(frame_count - 1):
            speed_rows.append(
                [
                    pair_idx + 1,
                    os.path.basename(csv_files[pair_idx]),
                    os.path.basename(csv_files[pair_idx + 1]),
                    self._to_csv_value(leftmost_y[pair_idx]),
                    self._to_csv_value(leftmost_y[pair_idx + 1]),
                    self._to_csv_value(delta_y[pair_idx]),
                    self._to_csv_value(speeds[pair_idx]),
                ]
            )

        valid_indices = np.where(np.isfinite(leftmost_y))[0]
        overall_speed = np.nan
        displacement = np.nan
        total_time = np.nan
        if valid_indices.size >= 2:
            first_idx = int(valid_indices[0])
            last_idx = int(valid_indices[-1])
            displacement = leftmost_y[first_idx] - leftmost_y[last_idx]
            total_time = (last_idx - first_idx) * frame_interval
            if total_time > 0:
                overall_speed = displacement * pixel_scale / total_time

        summary_rows = [
            [
                self._to_csv_value(displacement),
                self._to_csv_value(total_time),
                self._to_csv_value(overall_speed),
            ]
        ]

        position_path = os.path.join(output_folder, "speed_leftmost_position.csv")
        speed_path = os.path.join(output_folder, "speed_leftmost_series.csv")
        summary_path = os.path.join(output_folder, "speed_leftmost_summary.csv")

        self._save_csv(
            position_path,
            ["frame_index", "file_name", "leftmost_row", "leftmost_y"],
            position_rows,
        )
        self._save_csv(
            speed_path,
            [
                "frame_pair_index",
                "start_file",
                "end_file",
                "y_start",
                "y_end",
                "delta_y",
                "leftward_speed",
            ],
            speed_rows,
        )
        self._save_csv(
            summary_path,
            ["overall_displacement", "overall_time", "overall_leftward_speed"],
            summary_rows,
        )

        self.last_plot_payload = {
            "title": "最左传播点速度曲线",
            "x_label": "帧对索引",
            "y_label": "向左传播速度",
            "x_values": list(range(1, frame_count)),
            "series": [("leftmost_speed", speeds.tolist())],
        }

        self._log(f"已输出: {position_path}")
        self._log(f"已输出: {speed_path}")
        self._log(f"已输出: {summary_path}")
        self._log("算法2完成: 已生成最左传播点速度数据。")

    def on_compute_average_speed(self):
        prepared = self._prepare_common_inputs()
        if prepared is None:
            return

        csv_files = prepared["csv_files"]
        matrix = prepared["matrix"]
        frame_interval = prepared["frame_interval"]
        pixel_scale = prepared["pixel_scale"]
        output_folder = prepared["output_folder"]

        frame_count = len(csv_files)
        mean_y = np.full(frame_count, np.nan, dtype=float)
        valid_counts = np.zeros(frame_count, dtype=int)

        for frame_idx in range(frame_count):
            frame_values = matrix[frame_idx, :]
            valid_values = frame_values[np.isfinite(frame_values)]
            valid_counts[frame_idx] = int(valid_values.size)
            if valid_values.size > 0:
                mean_y[frame_idx] = float(np.mean(valid_values))

        delta_y, speeds = self._compute_pair_speed(mean_y, pixel_scale, frame_interval)

        position_rows = []
        for frame_idx, csv_path in enumerate(csv_files):
            position_rows.append(
                [
                    frame_idx + 1,
                    os.path.basename(csv_path),
                    valid_counts[frame_idx],
                    self._to_csv_value(mean_y[frame_idx]),
                ]
            )

        speed_rows = []
        for pair_idx in range(frame_count - 1):
            speed_rows.append(
                [
                    pair_idx + 1,
                    os.path.basename(csv_files[pair_idx]),
                    os.path.basename(csv_files[pair_idx + 1]),
                    self._to_csv_value(mean_y[pair_idx]),
                    self._to_csv_value(mean_y[pair_idx + 1]),
                    self._to_csv_value(delta_y[pair_idx]),
                    self._to_csv_value(speeds[pair_idx]),
                ]
            )

        valid_indices = np.where(np.isfinite(mean_y))[0]
        overall_speed = np.nan
        displacement = np.nan
        total_time = np.nan
        if valid_indices.size >= 2:
            first_idx = int(valid_indices[0])
            last_idx = int(valid_indices[-1])
            displacement = mean_y[first_idx] - mean_y[last_idx]
            total_time = (last_idx - first_idx) * frame_interval
            if total_time > 0:
                overall_speed = displacement * pixel_scale / total_time

        summary_rows = [
            [
                self._to_csv_value(displacement),
                self._to_csv_value(total_time),
                self._to_csv_value(overall_speed),
            ]
        ]

        position_path = os.path.join(output_folder, "speed_average_position.csv")
        speed_path = os.path.join(output_folder, "speed_average_series.csv")
        summary_path = os.path.join(output_folder, "speed_average_summary.csv")

        self._save_csv(
            position_path,
            ["frame_index", "file_name", "valid_row_count", "mean_y"],
            position_rows,
        )
        self._save_csv(
            speed_path,
            [
                "frame_pair_index",
                "start_file",
                "end_file",
                "mean_y_start",
                "mean_y_end",
                "delta_y",
                "leftward_speed",
            ],
            speed_rows,
        )
        self._save_csv(
            summary_path,
            ["overall_displacement", "overall_time", "overall_leftward_speed"],
            summary_rows,
        )

        self.last_plot_payload = {
            "title": "区间平均传播速度曲线",
            "x_label": "帧对索引",
            "y_label": "向左传播速度",
            "x_values": list(range(1, frame_count)),
            "series": [("average_speed", speeds.tolist())],
        }

        self._log(f"已输出: {position_path}")
        self._log(f"已输出: {speed_path}")
        self._log(f"已输出: {summary_path}")
        self._log("算法3完成: 已生成区间平均速度数据。")

    def on_plot_speed_curves(self):
        if not self.last_plot_payload:
            QMessageBox.warning(self, "提示", "请先运行任意一个速度算法，再绘图。")
            return
        dialog = SpeedPlotDialog(self.last_plot_payload, self)
        dialog.exec_()

    def _prepare_common_inputs(self) -> Optional[dict]:
        if not self.input_folder or not os.path.isdir(self.input_folder):
            QMessageBox.warning(self, "警告", "请先选择有效的CSV输入文件夹。")
            return None
        if not self.output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹。")
            return None

        row_start = int(self.row_start_spin.value())
        row_end = int(self.row_end_spin.value())
        if row_start > row_end:
            QMessageBox.warning(self, "警告", "起始行不能大于结束行。")
            return None

        frame_interval = float(self.frame_interval_spin.value())
        pixel_scale = float(self.pixel_scale_spin.value())
        if frame_interval <= 0:
            QMessageBox.warning(self, "警告", "帧间隔必须大于0。")
            return None
        if pixel_scale <= 0:
            QMessageBox.warning(self, "警告", "空间标定必须大于0。")
            return None

        csv_files = self._get_csv_files(self.input_folder)
        if len(csv_files) < 2:
            QMessageBox.warning(self, "警告", "CSV文件数量不足，至少需要2个文件。")
            return None

        os.makedirs(self.output_folder, exist_ok=True)

        valid_csv_files, rows, matrix = self._build_row_matrix(
            csv_files, row_start, row_end
        )
        if len(valid_csv_files) < 2:
            QMessageBox.warning(
                self, "警告", "有效CSV文件不足2个，请检查输入目录中的CSV格式。"
            )
            return None
        if matrix.size == 0:
            QMessageBox.warning(self, "警告", "指定行范围内没有可用数据。")
            return None

        self._log(f"读取CSV数量: {len(csv_files)} (有效: {len(valid_csv_files)})")
        self._log(f"行范围: {row_start} - {row_end} (共{len(rows)}行)")
        self._log(f"空间标定: {pixel_scale}, 帧间隔: {frame_interval}")

        return {
            "csv_files": valid_csv_files,
            "rows": rows,
            "matrix": matrix,
            "frame_interval": frame_interval,
            "pixel_scale": pixel_scale,
            "output_folder": self.output_folder,
        }

    def _get_csv_files(self, folder: str) -> List[str]:
        csv_paths = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".csv")
        ]
        csv_paths.sort(key=lambda p: self._natural_key(os.path.basename(p)))
        return csv_paths

    def _build_row_matrix(
        self, csv_files: List[str], row_start: int, row_end: int
    ) -> Tuple[List[str], List[int], np.ndarray]:
        rows = list(range(row_start, row_end + 1))
        valid_csv_files: List[str] = []
        matrix_rows: List[List[float]] = []

        for csv_file in csv_files:
            try:
                row_map = self._load_transition_csv(csv_file)
            except ValueError as e:
                self._log(f"跳过CSV(格式不匹配): {os.path.basename(csv_file)} | {e}")
                continue

            frame_row = [np.nan] * len(rows)
            for row_idx, row_no in enumerate(rows):
                if row_no in row_map:
                    frame_row[row_idx] = row_map[row_no]
            valid_csv_files.append(csv_file)
            matrix_rows.append(frame_row)

        if not matrix_rows:
            return valid_csv_files, rows, np.array([], dtype=float)

        matrix = np.asarray(matrix_rows, dtype=float)
        return valid_csv_files, rows, matrix

    def _load_transition_csv(self, csv_path: str) -> Dict[int, float]:
        row_map: Dict[int, float] = {}
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
            if not fieldnames or "x" not in fieldnames or "y" not in fieldnames:
                raise ValueError(f"CSV格式不符合要求: {csv_path}, 需要列名 X,Y")

            for line in reader:
                x_raw = (line.get("X") or line.get("x") or "").strip()
                y_raw = (line.get("Y") or line.get("y") or "").strip()
                if not x_raw:
                    continue
                try:
                    row_index = int(float(x_raw))
                except ValueError:
                    continue

                if not y_raw or y_raw.lower() in {"nan", "none"}:
                    row_map[row_index] = np.nan
                    continue
                try:
                    row_map[row_index] = float(y_raw)
                except ValueError:
                    row_map[row_index] = np.nan

        return row_map

    def _compute_pair_speed(
        self, y_values: np.ndarray, pixel_scale: float, frame_interval: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        pair_count = max(0, len(y_values) - 1)
        delta_y = np.full(pair_count, np.nan, dtype=float)
        speeds = np.full(pair_count, np.nan, dtype=float)

        for idx in range(pair_count):
            y_start = y_values[idx]
            y_end = y_values[idx + 1]
            if not (np.isfinite(y_start) and np.isfinite(y_end)):
                continue
            delta = y_start - y_end
            delta_y[idx] = delta
            speeds[idx] = delta * pixel_scale / frame_interval

        return delta_y, speeds

    def _save_csv(self, output_path: str, header: List[str], rows: List[List[object]]):
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def _natural_key(self, text: str):
        parts = re.split(r"(\d+)", text)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    def _to_csv_value(self, value):
        if value is None:
            return "nan"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "nan"
        if np.isfinite(numeric):
            return numeric
        return "nan"

    def _log(self, message: str):
        self.log_text.append(message)
