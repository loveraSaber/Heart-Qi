# 项目环境配置指南

## ✅ 当前状态

项目结构验证结果：
- **文件结构**: ✅ 100% 完整
- **配置加载**: ✅ 正常
- **数据模型**: ✅ 可用
- **模型注册表**: ✅ 工作中

**需要**: 安装依赖包

## 🔧 环境配置步骤

### 方式 1: 使用 venv（推荐）

```bash
# 1️⃣ 创建虚拟环境
cd c:\Users\qq152\Desktop\visualmodel
python -m venv venv

# 2️⃣ 激活虚拟环境（选择对应系统的命令）
# Windows CMD:
venv\Scripts\activate.bat

# Windows PowerShell:
venv\Scripts\Activate.ps1

# 3️⃣ 升级 pip
python -m pip install --upgrade pip

# 4️⃣ 安装依赖（选择对应的）

# 选项 A: 基础依赖（快速测试）
pip install fastapi uvicorn pydantic pydantic-settings numpy pandas pyyaml python-dotenv

# 选项 B: 完整依赖（包括深度学习）
pip install -r requirements.txt
```

### 方式 2: 使用 conda

```bash
# 1️⃣ 创建 conda 环境
conda create -n visualmodel python=3.10

# 2️⃣ 激活环境
conda activate visualmodel

# 3️⃣ 安装依赖
pip install -r requirements.txt
```

## 🧪 验证安装

安装完成后，再次运行测试：

```bash
python test_structure.py
```

预期所有 8 个测试都会通过。

## 📝 最小化依赖（如果想快速测试）

```bash
# 只安装基础包
pip install fastapi uvicorn pydantic pydantic-settings numpy pandas pyyaml python-dotenv redis celery

# 可以跳过的包（暂不使用）
# - torch / torchvision （如果暂不加载模型）
# - batch-face / py-feat （特征检测）
# - opencv-python （视频处理）
```

## 🚀 快速启动测试

### 1. 结构验证（无需任何 ML 包）
```bash
python test_structure.py
```

### 2. API 启动测试（需要 FastAPI）
```bash
# 安装最小依赖
pip install fastapi uvicorn

# 启动服务
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

然后访问: http://localhost:8000/docs

### 3. 完整应用测试（需要所有依赖 + Redis）
```bash
# 需要 Redis 运行
# redis-server

# 启动 Celery Worker
celery -A app.core.celery_app worker --loglevel=info

# 另一个终端启动 FastAPI
uvicorn server:app --reload
```

## 📋 依赖分类

### 必需（Core）
- fastapi, uvicorn, pydantic, pydantic-settings
- numpy, pandas
- pyyaml, python-dotenv

### 高优先级（High Priority）
- redis（缓存和任务队列）
- celery（异步任务）

### 可选（Optional）
- torch, torchvision, transformers（模型推理）
- opencv-python（视频处理）
- batch-face, py-feat（面部检测）

## ⚠️ GPU 支持

如果要使用 GPU（CUDA）：

```bash
# 1️⃣ 卸载 CPU 版本的 torch
pip uninstall torch

# 2️⃣ 安装 GPU 版本（适用于 CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 确认 CUDA 版本
python -c "import torch; print(torch.cuda.is_available())"
```

## 🔍 环境检查脚本

```python
# check_env.py
import sys

packages = {
    'fastapi': 'FastAPI',
    'pydantic': 'Pydantic',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'yaml': 'PyYAML',
    'redis': 'Redis',
    'celery': 'Celery',
    'torch': 'PyTorch',
    'cv2': 'OpenCV',
}

for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name} (缺失)")

# 检查 GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU 不可用，使用 CPU 模式")
except:
    print("⚠️  无法检查 GPU")
```

运行: `python check_env.py`

## 💡 常见问题

**Q: PyTorch 下载很慢？**
A: 使用清华镜像：`pip install torch -i https://mirrors.tsinghua.edu.cn/pypi/web/simple`

**Q: Redis 安装失败？**
A: Windows 用户可以用: `pip install redis-py` 而不是 `redis`

**Q: CUDA out of memory？**
A: 在 `app/config/env.py` 中降低 batch_size: `batch_size: int = 16`

## ✅ 下一步

1. [ ] 选择安装方式（venv 或 conda）
2. [ ] 安装依赖包
3. [ ] 运行 `python test_structure.py` 验证
4. [ ] 所有测试通过后，开始开发
