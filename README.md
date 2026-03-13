# 图像处理软件 - 用户安装说明

## 系统要求

### 基本要求（所有电脑）
- 操作系统：Windows 10/11 64位
- 内存：至少8GB RAM
- 磁盘空间：至少2GB可用空间

### GPU加速要求（可选）
- NVIDIA显卡（支持CUDA）
- 显存：建议8GB以上
- CUDA驱动：12.x版本

## 安装方式

### 方式一：直接运行exe文件（文件过大已删除）

1. **下载软件**
   - 获取 `IMG_Process_SW.exe` 文件

2. **首次运行**
   - 双击 `IMG_Process_SW.exe` 启动程序
   - 如果Windows提示"无法运行"，点击"更多信息" → "仍要运行"
   - 如果杀毒软件报警，请添加信任

3. **检查运行模式**
   - 程序启动后，主页会显示当前运行模式：
     - ✅ **GPU加速已启用**：检测到NVIDIA显卡，将使用GPU加速
     - ⚠️ **CPU模式运行**：未检测到GPU，将使用CPU处理

### 方式二：从源码运行

1. **安装Python**
   - 下载并安装 Python 3.10 或更高版本
   - 安装时勾选 "Add Python to PATH"

2. **下载源码**
   ```bash
   git clone https://github.com/kazeokamali/ImgProcessForLab.git
   ```

3. **创建虚拟环境**
   ```bash
   python -m venv venv310
   .\venv310\Scripts\activate
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

5. **运行程序**
   ```bash
   python main.py
   ```

## GPU加速配置（可选）

### 如果您有NVIDIA显卡

1. **检查显卡驱动**
   - 右键"此电脑" → "管理" → "设备管理器" → "显示适配器"
   - 确认有NVIDIA显卡

2. **安装CUDA驱动**
   - 访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
   - 下载并安装 CUDA Toolkit 12.x
   - 安装完成后重启电脑

3. **验证CUDA安装**
   - 打开命令提示符，输入：
     ```bash
     nvidia-smi
     ```
   - 如果显示显卡信息，说明CUDA已正确安装

4. **安装CuPy（GPU加速库）**
   ```bash
   pip install cupy-cuda12x
   ```

5. **重启程序**
   - 重新运行程序，主页应显示"GPU加速已启用"

## 性能对比

| 处理模式 | 100张图像处理时间 | 适用场景 |
|---------|----------------|---------|
| GPU加速 | 约30秒 | 大批量图像处理 |
| CPU模式 | 约3-5分钟 | 小批量或单张图像 |

## 常见问题

### 1. 程序无法启动
**解决方案：**
- 确认操作系统为Windows 10/11 64位
- 右键程序 → "属性" → "兼容性" → 勾选"以兼容模式运行"
- 关闭杀毒软件后重试

### 2. 提示缺少DLL文件
**解决方案：**
- 安装 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- 重启电脑后重试

### 3. GPU加速未启用
**解决方案：**
- 确认已安装NVIDIA显卡驱动
- 确认已安装CUDA Toolkit 12.x
- 确认已安装cupy-cuda12x（仅源码运行需要）
- 重启程序

### 4. 处理速度慢
**解决方案：**
- 如果使用CPU模式，这是正常现象
- 建议安装NVIDIA显卡并启用GPU加速
- 减少批处理图像数量

### 5. 内存不足错误
**解决方案：**
- 关闭其他程序释放内存
- 减少批处理图像数量
- 使用GPU加速（显存通常比内存大）

## 技术支持

如遇到其他问题，请提供以下信息：
- 操作系统版本
- 显卡型号（如有）
- CUDA版本（如有）
- 错误截图或错误信息

## 更新日志

### v1.0.0
- 支持CPU/GPU自适应运行
- 图像处理核心功能
- 黑线去除功能
- 文件重命名功能
- 图像信息查看和阈值划分功能
