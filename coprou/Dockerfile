FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        python3 python3-dev python3-pip \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
        libjpeg-dev libpng-dev \
        git wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && python -m pip install --upgrade pip setuptools wheel

# PyTorch (как в README)
RUN pip install --no-cache-dir \
    torch==2.7.0+cu118 \
    torchvision==0.22.0+cu118 \
    torchaudio==2.7.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# xformers
RUN pip install --no-cache-dir \
    xformers==0.0.30 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Копируем requirements внутрь образа
COPY requirements.txt .

# Устанавливаем зависимости проекта + фикс OpenCV
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir opencv-python-headless==4.8.1.78

ENV PYTHONPATH="/workspace/CoProU"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/CoProU
CMD ["bash"]