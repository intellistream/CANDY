#!/bin/bash

# Detect if running in WSL
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    echo "Running in WSL"
else
    echo "This script is intended for WSL only."
    exit 1
fi

# Detect if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    GPU_AVAILABLE=true
else
    echo "No NVIDIA GPU detected"
    GPU_AVAILABLE=false
fi

# Update system and install essential packages and dependencies
sudo apt-get update
sudo apt-get install -y --allow-change-held-packages \
    sudo \
    build-essential \
    cmake \
    curl \
    unzip \
    liblapack-dev \
    libblas-dev \
    libboost-dev \
    rsync \
    nano \
    graphviz \
    python3 \
    python3-pip \
    gdb \
    liblog4cxx-dev \
    git \
    libcupti-dev \
    nvidia-cuda-toolkit

# Add CUDA to PATH if installed
CUDA_PATH="/usr"
if [ -d "$CUDA_PATH" ]; then
    echo "CUDA found at: $CUDA_PATH"
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export CUDACXX=$CUDA_PATH/bin/nvcc
    echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDACXX=$CUDA_PATH/bin/nvcc" >> ~/.bashrc
    source ~/.bashrc
else
    echo "CUDA installation not found. Please verify the installation."
fi

# Check if pip3 is available, and install if it's missing
if ! command -v pip3 &> /dev/null; then
    echo "pip3 could not be found. Installing python3-pip..."
    sudo apt-get install -y python3-pip
fi

# Install Python dependencies
if [ "$GPU_AVAILABLE" = true ]; then
    # Install CUDA version of PyTorch if GPU is available
    pip3 install torch==2.4.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU version of PyTorch if no GPU is available
    pip3 install torch==2.4.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo "WSL setup complete."
