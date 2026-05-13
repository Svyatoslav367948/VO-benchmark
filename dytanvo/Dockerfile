FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build \
        python3.8 python3.8-dev python3-pip \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
        libjpeg-dev libpng-dev \
        git wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ── PyTorch 1.11.0 + CUDA 11.3
RUN pip install \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# ── Остальные зависимости
RUN pip install \
    numpy scipy \
    "opencv-python==4.5.5.64" \
    "opencv-contrib-python==4.5.5.64" \
    Pillow imageio scikit-image \
    "kornia==0.5.3" einops \
    matplotlib pandas tqdm \
    pyquaternion transforms3d \
    pyyaml colorama \
    evo

RUN pip install cupy-cuda113

# ── DCNv2: клонируем и патчим для PyTorch 1.11.0 + CUDA 11.3
RUN git clone https://github.com/MatthewHowe/DCNv2.git /opt/DCNv2

COPY dcn_v2_cuda_patched.cu /opt/DCNv2/DCN/src/cuda/dcn_v2_cuda.cu

RUN sed -i 's|#include <TH/TH.h>||g' \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_cpu.cpp \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_im2col_cpu.cpp \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_psroi_pooling_cpu.cpp && \
    sed -i \
        -e 's|#include <THC/THC\.h>||g' \
        -e 's|#include <THC/THCAtomics\.cuh>||g' \
        -e 's|#include <THC/THCDeviceUtils\.cuh>||g' \
        /opt/DCNv2/DCN/src/cuda/dcn_v2_psroi_pooling_cuda.cu \
        /opt/DCNv2/DCN/src/cuda/dcn_v2_im2col_cuda.cu && \
    sed -i 's|TORCH_CHECK_ARG(\(.*\), [0-9]\+, \(.*\))|TORCH_CHECK(\1, \2)|g' \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_cpu.cpp && \
    sed -i 's|THArgCheck(\(.*\), [0-9]\+, \(.*\));|TORCH_CHECK(\1, \2);|g' \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_cpu.cpp && \
    sed -i 's|\.data<\(.*\)>()|.data_ptr<\1>()|g' \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_cpu.cpp \
        /opt/DCNv2/DCN/src/cpu/dcn_v2_psroi_pooling_cpu.cpp

RUN cd /opt/DCNv2 && \
    TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    pip install . --no-build-isolation

ENV PYTHONPATH="/workspace/DytanVO"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/DytanVO
CMD ["bash"]