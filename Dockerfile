FROM python:3.10-slim

WORKDIR /app

# ── 1. apt 换阿里云源 ──────────────────────────────────────────
RUN echo "deb https://mirrors.aliyun.com/debian/ trixie main contrib non-free non-free-firmware" > /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian/ trixie-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian-security trixie-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        curl \
        ca-certificates \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        liblapack-dev \
        libblas-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# ── 2. pip 换阿里云源 ──────────────────────────────────────────
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 3. 安装基础依赖 ────────────────────────────────────────────
COPY requirements-slim.txt .
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 600 \
    -r requirements-slim.txt

# ── 4. 安装 PyTorch CPU ────────────────────────────────────────
# 阿里云为主源，官方 whl 兜底
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 1000 \
    --trusted-host mirrors.aliyun.com \
    --trusted-host download.pytorch.org \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.1+cpu \
    torchvision==0.16.1+cpu
RUN pip install --no-cache-dir flower==2.0.1
# ── 5. 安装 transformers + opencv ─────────────────────────────
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 600 \
    transformers==4.35.0 \
    opencv-python-headless==4.8.1.78

# ── 6. 安装可选深度学习依赖 ────────────────────────────────────
# 这两个包依赖复杂，单独一层，失败可单独排查
RUN pip install --no-cache-dir \
    --retries 3 \
    --timeout 600 \
    batch-face==1.5.0 \
    py-feat==0.6.2
# 强制修复 opencv，防止被上面的包覆盖为普通版
RUN pip install --no-cache-dir --force-reinstall \
    opencv-python-headless==4.8.1.78
#   强制固定 numpy 版本，防止被其他包覆盖 ──────────────────
RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.24.3 \
    scipy==1.11.4
# ── 7. 复制项目代码（最后一步，保护前面的缓存层）──────────────
COPY . .

# ── 8. 启动 ───────────────────────────────────────────────────
EXPOSE 9096

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9096/docs || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9096", "--log-level", "info"]