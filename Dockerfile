FROM python:3.10-slim

WORKDIR /app

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip 和基础工具
RUN pip install --upgrade pip setuptools wheel

# 3. 复制依赖文件
COPY requirements-core.txt requirements-ml.txt requirements-optional.txt ./

# 4. 安装核心依赖（官方源）
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 1000 \
    -r requirements-core.txt

# 5. 安装 PyTorch（从官方源，更稳定）
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 1000 \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements-ml.txt || echo "⚠️  PyTorch 安装失败，尝试继续"

# 6. 安装可选依赖（失败继续）
RUN pip install --no-cache-dir \
    --retries 3 \
    --timeout 600 \
    -r requirements-optional.txt || echo "⚠️  可选依赖安装失败，继续启动应用"

# 7. 复制项目代码
COPY . .

# 8. 暴露端口
EXPOSE 9096

# 9. 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9096/docs || exit 1

# 10. 启动命令
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9096", "--log-level", "info"]