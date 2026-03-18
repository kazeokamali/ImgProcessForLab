# 用户安装与环境配置指南（双入口）

更新时间：2026-03-18

## 1. 项目有两个入口

本项目不是单一程序，而是两个入口：

- `main.py`：综合处理主程序  
  功能包括图像处理、黑线去除、环伪影后处理、多点平场、坏点掩膜、超视野CT模拟、波速分析、文件重命名。
- `main_recon.py`：独立重构程序  
  功能为 CBCT 重构（FDK/SIRT/CGLS 等）。

建议把这两个入口都交付给用户，并在文档中明确用途。

## 2. 系统与硬件要求

基础要求（必须）：

- Windows 10/11 64位
- Python 3.10
- 内存 16GB 及以上（建议）
- 可用磁盘空间 10GB 及以上（数据量大时建议更多）

GPU加速要求（可选）：

- NVIDIA 显卡
- 已安装可用驱动（`nvidia-smi` 可正常返回）
- 若使用 CuPy：CUDA 12.x 对应版本（`cupy-cuda12x`）
- 若使用 ASTRA CUDA：建议用 Conda 安装 `astra-toolbox`

## 3. 推荐环境方案

推荐采用“双环境”：

- `venv310`：运行主程序 `main.py`（以及重构界面 UI）
- `astra_env`（Conda）：用于重构计算后端（ASTRA）

这样做的好处是：UI依赖和 ASTRA/CUDA 依赖分离，升级与排错更稳定。

## 4. 安装步骤（主程序环境：venv310）

在项目根目录执行：

```powershell
python -m venv venv310
.\venv310\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

说明：

- 该环境用于运行 `main.py`
- 也可以运行 `main_recon.py` 的界面部分

## 5. 安装步骤（重构后端环境：astra_env，可选但推荐）

### 5.1 创建环境

```powershell
conda create -n astra_env python=3.10 -y
conda activate astra_env
```

### 5.2 安装 ASTRA

```powershell
conda install -c astra-toolbox astra-toolbox -y
```

### 5.3 安装重构运行所需基础包

```powershell
pip install numpy scipy tifffile imageio pyqt5
```

如你的 `astra_env` 已包含这些包，可跳过。

## 6. 运行方式（两个入口）

### 6.1 运行综合处理主程序

```powershell
.\venv310\Scripts\activate
python main.py
```

### 6.2 运行独立重构程序

方式A（推荐）：

- 用 `venv310` 启动 `main_recon.py` 的界面
- 在重构页面里把“工作进程 Python”设置为 `astra_env` 的 `python.exe`

```powershell
.\venv310\Scripts\activate
python main_recon.py
```

方式B（可选）：

- 直接在 `astra_env` 中运行 `main_recon.py`（前提是该环境已安装 PyQt5）

```powershell
conda activate astra_env
python main_recon.py
```

## 7. 安装后检查

### 7.1 GPU驱动检查

```powershell
nvidia-smi
```

### 7.2 主程序检查

- 启动 `main.py`
- 首页会显示当前状态（GPU加速已启用 / CPU模式运行）

### 7.3 重构检查

- 启动 `main_recon.py`
- 执行一组小数据重构，检查日志中的后端信息（如 `astra_fdk_cuda`）

## 8. 常见问题

### 8.1 `No module named cupy`

说明：主程序 GPU 功能依赖 CuPy。  
处理：

```powershell
.\venv310\Scripts\activate
pip install cupy-cuda12x
```

若不装 CuPy，程序会回退 CPU，功能仍可用。

### 8.2 重构时报 ASTRA 不可用

说明：ASTRA 未安装到当前解释器。  
处理：

- 把重构页面中的“工作进程 Python”切到 `astra_env\python.exe`
- 或在当前环境安装 `astra-toolbox`

### 8.3 RAW 尺寸不匹配

说明：RAW 文件宽高设置与真实尺寸不一致。  
处理：在对应页面调整 `RAW 宽度/高度` 参数。

### 8.4 启动即闪退或缺 DLL

处理：

- 安装 Visual C++ Redistributable (x64)
- 确认 Python、依赖、驱动版本匹配
- 用终端运行查看报错而不是直接双击

## 9. 给其他用户的建议


建议双环境：

- `venv310` 创建与安装脚本
- `astra_env` 创建与安装脚本

