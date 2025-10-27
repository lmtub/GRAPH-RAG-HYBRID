FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Cài gói hệ thống (có Java cho Joern; bỏ nếu chưa cần)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential \
    openjdk-17-jre-headless \
 && rm -rf /var/lib/apt/lists/*

 # --- Joern CLI (pin version) ---
RUN curl -L -o /tmp/joern.zip https://github.com/joernio/joern/releases/download/v1.1.230/joern-cli.zip \
 && unzip /tmp/joern.zip -d /opt/joern && rm /tmp/joern.zip \
 && ln -s /opt/joern/joern-cli/joern /usr/local/bin/joern \
 && ln -s /opt/joern/joern-cli/bin/joern-parse /usr/local/bin/joern-parse \
 && ln -s /opt/joern/joern-cli/bin/joern-export /usr/local/bin/joern-export

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
CMD ["python", "src/train.py"]
