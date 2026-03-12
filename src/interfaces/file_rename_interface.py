from dataclasses import dataclass
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
)
from qfluentwidgets import FluentIcon


@dataclass
class FileInfo:
    path: str
    name: str
    ext: str
    full_name: str

    @classmethod
    def from_path(cls, path: str) -> "FileInfo":
        p = Path(path)
        return cls(
            path=str(p.absolute()),
            name=p.stem,
            ext=p.suffix,
            full_name=p.name,
        )

    def get_new_path(self, new_name: str) -> str:
        p = Path(self.path)
        new_path = p.parent / new_name
        return str(new_path.absolute())


class FileRenameInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fileRenameInterface")
        self.files: list[FileInfo] = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title_layout = QHBoxLayout()
        title_label = QLabel("文件批量重命名", self)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; margin-bottom: 20px;")
        title_layout.addWidget(title_label)
        layout.addLayout(title_layout)

        operation_group = QGroupBox("重命名操作")
        operation_layout = QVBoxLayout()

        operation_type_layout = QHBoxLayout()
        operation_type_label = QLabel("操作类型:")
        operation_type_label.setStyleSheet("font-weight: bold;")
        operation_type_layout.addWidget(operation_type_label)

        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "查找替换",
            "查找删除",
            "格式化",
            "插入",
            "自定义脚本"
        ])
        self.operation_combo.currentIndexChanged.connect(self.on_operation_changed)
        operation_type_layout.addWidget(self.operation_combo)

        operation_layout.addLayout(operation_type_layout)

        self.case_sensitive_checkbox = QPushButton("区分大小写")
        self.case_sensitive_checkbox.setCheckable(True)
        self.case_sensitive_checkbox.setStyleSheet("font-size: 12px; padding: 5px;")
        operation_layout.addWidget(self.case_sensitive_checkbox)

        self.use_regex_checkbox = QPushButton("使用正则表达式")
        self.use_regex_checkbox.setCheckable(True)
        self.use_regex_checkbox.setStyleSheet("font-size: 12px; padding: 5px;")
        operation_layout.addWidget(self.use_regex_checkbox)

        operation_group.setLayout(operation_layout)
        layout.addWidget(operation_group)

        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        layout.addWidget(self.params_container)

        files_group = QGroupBox("文件列表")
        files_layout = QVBoxLayout()

        button_layout = QHBoxLayout()

        add_files_btn = QPushButton("添加文件")
        add_files_btn.clicked.connect(self.add_files)
        button_layout.addWidget(add_files_btn)

        add_folder_btn = QPushButton("添加文件夹")
        add_folder_btn.clicked.connect(self.add_folder)
        button_layout.addWidget(add_folder_btn)

        clear_btn = QPushButton("清空列表")
        clear_btn.clicked.connect(self.clear_files)
        button_layout.addWidget(clear_btn)

        remove_selected_btn = QPushButton("移除选中")
        remove_selected_btn.clicked.connect(self.remove_selected)
        button_layout.addWidget(remove_selected_btn)

        files_layout.addLayout(button_layout)

        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels([
            "原文件名", 
            "新文件名", 
            "文件路径", 
            "状态"
        ])

        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setAlternatingRowColors(True)

        files_layout.addWidget(self.file_table)
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        action_layout = QHBoxLayout()

        preview_btn = QPushButton("预览重命名")
        preview_btn.clicked.connect(self.preview_rename)
        preview_btn.setStyleSheet("background-color: #0078d4; color: white; padding: 10px;")
        action_layout.addWidget(preview_btn)

        apply_btn = QPushButton("应用重命名")
        apply_btn.clicked.connect(self.apply_rename)
        apply_btn.setStyleSheet("background-color: #28a745; color: white; padding: 10px;")
        action_layout.addWidget(apply_btn)

        layout.addLayout(action_layout)
        layout.addStretch(1)

        self.on_operation_changed(0)

    def on_operation_changed(self, index):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        operation = self.operation_combo.currentText()

        if operation == "查找替换":
            self.setup_replace_ui()
        elif operation == "查找删除":
            self.setup_delete_ui()
        elif operation == "格式化":
            self.setup_format_ui()
        elif operation == "插入":
            self.setup_insert_ui()
        elif operation == "自定义脚本":
            self.setup_script_ui()

    def setup_replace_ui(self):
        params_group = QGroupBox("替换参数")
        params_layout = QVBoxLayout()

        search_layout = QHBoxLayout()
        search_label = QLabel("查找:")
        search_label.setStyleSheet("font-weight: bold;")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入要查找的文本...")
        search_layout.addWidget(self.search_input)
        search_layout.addStretch(1)

        replace_layout = QHBoxLayout()
        replace_label = QLabel("替换为:")
        replace_label.setStyleSheet("font-weight: bold;")
        replace_layout.addWidget(replace_label)

        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("输入替换文本...")
        replace_layout.addWidget(self.replace_input)
        replace_layout.addStretch(1)

        params_layout.addLayout(search_layout)
        params_layout.addLayout(replace_layout)
        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)

    def setup_delete_ui(self):
        params_group = QGroupBox("删除参数")
        params_layout = QVBoxLayout()

        search_layout = QHBoxLayout()
        search_label = QLabel("查找:")
        search_label.setStyleSheet("font-weight: bold;")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入要删除的文本...")
        search_layout.addWidget(self.search_input)
        search_layout.addStretch(1)

        params_layout.addLayout(search_layout)
        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)

    def setup_format_ui(self):
        params_group = QGroupBox("格式化参数")
        params_layout = QVBoxLayout()

        template_layout = QHBoxLayout()
        template_label = QLabel("模板:")
        template_label.setStyleSheet("font-weight: bold;")
        template_layout.addWidget(template_label)

        self.template_input = QLineEdit()
        self.template_input.setPlaceholderText("输入模板，如: tomo_#")
        template_layout.addWidget(self.template_input)
        template_layout.addStretch(1)

        params_layout.addLayout(template_layout)

        bit_layout = QHBoxLayout()
        bit_label = QLabel("位数:")
        bit_label.setStyleSheet("font-weight: bold;")
        bit_layout.addWidget(bit_label)

        self.bit_spinbox = QSpinBox()
        self.bit_spinbox.setRange(1, 10)
        self.bit_spinbox.setValue(4)
        bit_layout.addWidget(self.bit_spinbox)
        bit_layout.addStretch(1)

        start_layout = QHBoxLayout()
        start_label = QLabel("开始数字:")
        start_label.setStyleSheet("font-weight: bold;")
        start_layout.addWidget(start_label)

        self.start_spinbox = QSpinBox()
        self.start_spinbox.setRange(0, 999999)
        self.start_spinbox.setValue(0)
        start_layout.addWidget(self.start_spinbox)
        start_layout.addStretch(1)

        suffix_layout = QHBoxLayout()
        suffix_label = QLabel("后缀:")
        suffix_label.setStyleSheet("font-weight: bold;")
        suffix_layout.addWidget(suffix_label)

        self.suffix_input = QLineEdit()
        self.suffix_input.setPlaceholderText("输入后缀，如: .tiff")
        suffix_layout.addWidget(self.suffix_input)
        suffix_layout.addStretch(1)

        params_layout.addLayout(bit_layout)
        params_layout.addLayout(start_layout)
        params_layout.addLayout(suffix_layout)

        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)

    def setup_insert_ui(self):
        params_group = QGroupBox("插入参数")
        params_layout = QVBoxLayout()

        position_layout = QHBoxLayout()
        position_label = QLabel("插入位置:")
        position_label.setStyleSheet("font-weight: bold;")
        position_layout.addWidget(position_label)

        self.position_group = QButtonGroup()

        self.start_radio = QRadioButton("开头")
        self.start_radio.setChecked(True)
        self.position_group.addButton(self.start_radio)
        position_layout.addWidget(self.start_radio)

        self.end_radio = QRadioButton("结尾")
        self.position_group.addButton(self.end_radio)
        position_layout.addWidget(self.end_radio)

        self.middle_radio = QRadioButton("中间")
        self.position_group.addButton(self.middle_radio)
        position_layout.addWidget(self.middle_radio)
        position_layout.addStretch(1)

        params_layout.addLayout(position_layout)

        content_layout = QHBoxLayout()
        content_label = QLabel("插入内容:")
        content_label.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(content_label)

        self.insert_content_input = QLineEdit()
        self.insert_content_input.setPlaceholderText("输入要插入的内容...")
        content_layout.addWidget(self.insert_content_input)
        content_layout.addStretch(1)

        params_layout.addLayout(content_layout)

        regex_layout = QHBoxLayout()
        regex_label = QLabel("正则表达式(仅中间插入需要):")
        regex_label.setStyleSheet("font-weight: bold;")
        regex_layout.addWidget(regex_label)

        self.insert_regex_input = QLineEdit()
        self.insert_regex_input.setPlaceholderText("输入正则表达式，如: \\d+$")
        self.insert_regex_input.setEnabled(False)
        regex_layout.addWidget(self.insert_regex_input)
        regex_layout.addStretch(1)

        params_layout.addLayout(regex_layout)

        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)

        self.start_radio.toggled.connect(self.on_position_changed)
        self.end_radio.toggled.connect(self.on_position_changed)
        self.middle_radio.toggled.connect(self.on_position_changed)

    def setup_script_ui(self):
        params_group = QGroupBox("自定义脚本")
        params_layout = QVBoxLayout()

        script_layout = QVBoxLayout()
        script_label = QLabel("Python脚本:")
        script_label.setStyleSheet("font-weight: bold;")
        script_layout.addWidget(script_label)

        self.script_input = QLineEdit()
        self.script_input.setPlaceholderText("输入Python代码...")
        self.script_input.setPlaceholderText("result = name.upper() + '_' + str(index)")
        script_layout.addWidget(self.script_input)

        params_layout.addLayout(script_layout)

        hint_label = QLabel("可用变量: name(文件名), ext(扩展名), index(索引), full_name(全名), path(路径)")
        hint_label.setStyleSheet("color: gray; font-size: 11px; margin-top: 5px;")
        params_layout.addWidget(hint_label)

        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)

    def on_position_changed(self, checked):
        self.insert_regex_input.setEnabled(self.middle_radio.isChecked())

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", "", "所有文件 (*.*)"
        )
        if files:
            for file in files:
                file_info = FileInfo.from_path(file)
                self.add_file_to_table(file_info)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择文件夹", ""
        )
        if folder:
            self.add_files_from_folder(folder)

    def add_files_from_folder(self, folder: str):
        from pathlib import Path
        folder_path = Path(folder)
        for file in folder_path.iterdir():
            if file.is_file():
                file_info = FileInfo.from_path(str(file))
                self.add_file_to_table(file_info)

    def add_file_to_table(self, file_info: FileInfo):
        if file_info not in self.files:
            self.files.append(file_info)
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)

            self.file_table.setItem(row, 0, QTableWidgetItem(file_info.full_name))
            self.file_table.setItem(row, 1, QTableWidgetItem(""))
            self.file_table.setItem(row, 2, QTableWidgetItem(file_info.path))
            self.file_table.setItem(row, 3, QTableWidgetItem("等待处理"))

    def clear_files(self):
        self.files.clear()
        self.file_table.setRowCount(0)

    def remove_selected(self):
        selected_rows = set()
        for item in self.file_table.selectedItems():
            selected_rows.add(item.row())

        for row in sorted(selected_rows, reverse=True):
            if row < len(self.files):
                self.files.pop(row)
            self.file_table.removeRow(row)

    def preview_rename(self):
        if not self.files:
            QMessageBox.warning(self, "警告", "请先添加文件！")
            return

        operation = self.operation_combo.currentText()
        case_sensitive = self.case_sensitive_checkbox.isChecked()
        use_regex = self.use_regex_checkbox.isChecked()

        try:
            for row in range(self.file_table.rowCount()):
                if row < len(self.files):
                    file_info = self.files[row]
                    new_name = self.apply_operation(
                        file_info, 
                        operation, 
                        case_sensitive, 
                        use_regex
                    )
                    self.file_table.setItem(row, 1, QTableWidgetItem(new_name))
                    self.file_table.setItem(row, 3, QTableWidgetItem("预览"))

            QMessageBox.information(self, "成功", "预览完成！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预览失败: {str(e)}")

    def apply_rename(self):
        if not self.files:
            QMessageBox.warning(self, "警告", "请先添加文件！")
            return

        reply = QMessageBox.question(
            self, "确认",
            "确定要重命名这些文件吗？此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            operation = self.operation_combo.currentText()
            case_sensitive = self.case_sensitive_checkbox.isChecked()
            use_regex = self.use_regex_checkbox.isChecked()

            success_count = 0
            fail_count = 0

            try:
                for row in range(self.file_table.rowCount()):
                    if row < len(self.files):
                        file_info = self.files[row]
                        new_name = self.apply_operation(
                            file_info, 
                            operation, 
                            case_sensitive, 
                            use_regex
                        )
                        
                        new_path = file_info.get_new_path(new_name)
                        
                        try:
                            Path(file_info.path).rename(new_path)
                            success_count += 1
                            self.file_table.setItem(row, 3, QTableWidgetItem("成功"))
                        except Exception as e:
                            fail_count += 1
                            self.file_table.setItem(row, 3, QTableWidgetItem(f"失败: {str(e)}"))

                QMessageBox.information(
                    self, 
                    "完成", 
                    f"重命名完成！\n成功: {success_count}\n失败: {fail_count}"
                )

            except Exception as e:
                QMessageBox.critical(self, "错误", f"重命名失败: {str(e)}")

    def apply_operation(self, file_info: FileInfo, operation: str, 
                   case_sensitive: bool, use_regex: bool) -> str:
        name = file_info.name
        ext = file_info.ext

        if operation == "查找替换":
            search_text = self.search_input.text()
            replace_text = self.replace_input.text()
            if search_text:
                if use_regex:
                    import re
                    flags = 0 if case_sensitive else re.IGNORECASE
                    pattern = re.compile(search_text, flags)
                    name = pattern.sub(replace_text, name)
                else:
                    if case_sensitive:
                        name = name.replace(search_text, replace_text)
                    else:
                        name = name.replace(search_text.lower(), replace_text).replace(
                            search_text.upper(), replace_text
                        )
        
        elif operation == "查找删除":
            search_text = self.search_input.text()
            if search_text:
                if use_regex:
                    import re
                    flags = 0 if case_sensitive else re.IGNORECASE
                    pattern = re.compile(search_text, flags)
                    name = pattern.sub("", name)
                else:
                    if case_sensitive:
                        name = name.replace(search_text, "")
                    else:
                        name = name.replace(search_text.lower(), "").replace(
                            search_text.upper(), ""
                        )
        
        elif operation == "格式化":
            template = self.template_input.text()
            bit = self.bit_spinbox.value()
            start_num = self.start_spinbox.value()
            suffix = self.suffix_input.text()
            
            index = self.files.index(file_info)
            name = template.replace("#", str(start_num + index).zfill(bit))
            
            if suffix:
                ext = suffix
        
        elif operation == "插入":
            content = self.insert_content_input.text()
            if content:
                if self.start_radio.isChecked():
                    name = content + name
                elif self.end_radio.isChecked():
                    name = name + content
                elif self.middle_radio.isChecked():
                    regex = self.insert_regex_input.text()
                    if regex:
                        import re
                        match = re.search(regex, name)
                        if match:
                            pos = match.end()
                            name = name[:pos] + content + name[pos:]
        
        elif operation == "自定义脚本":
            pass

        return name + ext
