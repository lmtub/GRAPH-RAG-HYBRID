FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Gói hệ thống (có Java + unzip cho Joern)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential \
    openjdk-17-jre-headless unzip \
 && rm -rf /var/lib/apt/lists/*

# --- Cài Joern 4.x bằng installer chính thức ---
# Có thể pin version nếu muốn: thêm --version v4.0.440
RUN curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh \
  | bash -s -- --install-dir /opt/joern

# Đưa CLI vào PATH
ENV PATH="/opt/joern/joern-cli:${PATH}"

# (Optional) Kiểm tra nhanh giúp cache build fail sớm nếu lỗi
RUN joern-parse --help >/dev/null && joern-export --help >/dev/null

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ✅ Cài đúng phiên bản PyTorch + CUDA + PyTorch Geometric
RUN pip install --no-cache-dir --force-reinstall --upgrade \
    torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html && \
    pip install --no-cache-dir torch-geometric
    
COPY . .
CMD ["python", "src/train.py"]
