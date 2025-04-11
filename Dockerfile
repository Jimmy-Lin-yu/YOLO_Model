# 使用官方 PyTorch 支援 GPU 的映像
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# 安裝基本工具
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    nano \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0

# 建立工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --upgrade pip && pip install -r requirements.txt

# 將目前資料夾內容複製進去（可選）
# COPY . .

CMD ["bash"]
