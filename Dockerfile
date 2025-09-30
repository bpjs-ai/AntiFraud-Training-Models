# Dockerfile untuk BPJS AntiFraud Training Models
# Berdasarkan contoh Dockerfile.example dengan penyesuaian untuk training pipeline

# Gunakan base image Ubuntu 20.04
FROM ubuntu:20.04

# Hindari prompt interaktif dan set locale/timezone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Jakarta \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependensi sistem dasar
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates \
    wget \
    curl \
    sudo \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Tambahkan PPA untuk Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgeos-dev \
    ccache \
    git \
    libzbar0 \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip untuk Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Upgrade pip, setuptools, wheel
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Arahkan python3 ke python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy seluruh source code
COPY . .

# Buat user non-root
RUN useradd -m -s /bin/bash appuser && \
    echo 'appuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Buat direktori yang dibutuhkan & beri izin
RUN mkdir -p \
        /home/appuser/.config/Ultralytics \
        /app/dataset \
        /app/outputs \
        /app/config \
        /app/logs \
        /app/models \
        /app/temp && \
    SITE_PACKAGES_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    chown -R appuser:appuser /home/appuser /app "$SITE_PACKAGES_PATH"

# Gunakan user non-root
USER appuser

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('PyTorch available:', torch.cuda.is_available())" || exit 1

# Default command untuk training classifier dengan parameter yang dapat dikustomisasi
CMD ["python3", "main.py", "--pipeline", "classifier", "--epochs", "10", "--batch_size", "32", "--num_workers", "4", "--lr_bert", "2e-5", "--lr_resnet", "1e-4", "--lr_heads", "1e-3"]

# Labels untuk metadata
LABEL maintainer="BPJS AntiFraud" \
      version="1.0.0" \
      description="BPJS AntiFraud Training Models - ML Pipeline untuk Classifier dan Signature Detection"
