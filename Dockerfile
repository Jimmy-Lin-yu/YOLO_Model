# 使用 CUDA 12.8 + cuDNN 支援 Blackwell 架構（sm_120）
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 安裝基本工具與 Python
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    nano \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 建立工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 套件（包含對 Blackwell 的 PyTorch Nightly）
RUN pip3 install --upgrade pip && \
    pip3 uninstall -y torch torchvision torchaudio && \
    pip3 install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip3 install -r requirements.txt

# （可選）複製整個專案到容器
# COPY . .

# 進入 bash
CMD ["bash"]
