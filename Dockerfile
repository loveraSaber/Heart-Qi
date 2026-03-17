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

# ── 4. 安装 PyTorch GPU (CUDA 11.8) ───────────────────────────
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 1000 \
    --trusted-host mirrors.aliyun.com \
    --trusted-host download.pytorch.org \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118

# ── 5. 安装 flower ─────────────────────────────────────────────
RUN pip install --no-cache-dir flower==2.0.1

# ── 6. 安装 transformers + opencv ─────────────────────────────
RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 600 \
    transformers==4.35.0 \
    opencv-python-headless==4.8.1.78

# ── 7. 安装可选深度学习依赖 ────────────────────────────────────
RUN pip install --no-cache-dir \
    --retries 3 \
    --timeout 600 \
    batch-face==1.5.0 \
    py-feat==0.6.2

# ── 8. 强制修复版本，防止被覆盖 ───────────────────────────────
RUN pip install --no-cache-dir --force-reinstall \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3 \
    scipy==1.11.4
RUN pip install --no-cache-dir omegaconf
# ── 9. 复制项目代码 ────────────────────────────────────────────
COPY . .

# ── 10. 启动 ──────────────────────────────────────────────────
EXPOSE 9096

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9096/docs || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9096", "--log-level", "info"]