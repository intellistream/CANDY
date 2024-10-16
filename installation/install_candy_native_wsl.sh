#!/bin/bash

# Detect if running in WSL
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    echo "Running in WSL"
    IS_WSL=true
else
    echo "Running on native Ubuntu"
    IS_WSL=false
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
    openssh-server \
    gdb \
    liblog4cxx-dev \
    git \
    libcupti-dev

# Add NVIDIA repository for cuDNN and CUDA
if [ "$IS_WSL" = true ]; then
    # For WSL, install CUDA toolkit directly
    sudo apt-get install -y nvidia-cuda-toolkit
    CUDA_PATH="/usr"
else
    # For native Ubuntu, add the NVIDIA repository and install CUDA
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    wget https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda libcudnn8 libcudnn8-dev
    CUDA_PATH=$(ls -d /usr/local/cuda-*/ 2>/dev/null | tail -n 1)
fi

# Add CUDA to PATH if installed
if [ -n "$CUDA_PATH" ]; then
    echo "CUDA found at: $CUDA_PATH"
    if [ "$IS_WSL" = true ]; then
        # For WSL, adjust PATH and LD_LIBRARY_PATH
        export PATH=$CUDA_PATH/bin:$PATH
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        export CUDACXX=$CUDA_PATH/bin/nvcc
        echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH" >> ~/.bashrc
        echo "export CUDACXX=$CUDA_PATH/bin/nvcc" >> ~/.bashrc
    else
        export PATH=${CUDA_PATH}/bin:$PATH
        export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
        export CUDACXX=${CUDA_PATH}/bin/nvcc
        echo "export PATH=${CUDA_PATH}/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        echo "export CUDACXX=${CUDA_PATH}/bin/nvcc" >> ~/.bashrc
    fi
    source ~/.bashrc
else
    echo "CUDA installation not found. Please verify the installation."
fi

# Check if pip3 is available, and install if it's missing
if ! command -v pip3 &> /dev/null; then
    echo "pip3 could not be found. Installing python3-pip..."
    sudo apt-get install -y python3-pip
fi

# Ensure pip3 is in PATH
if ! command -v pip3 &> /dev/null; then
    echo "pip3 installation failed or pip3 is not in PATH. Attempting to locate pip3..."
    pip3_path=$(find / -name pip3 2>/dev/null | head -n 1)
    if [ -n "$pip3_path" ]; then
        echo "Adding pip3 to PATH..."
        export PATH=$PATH:$(dirname $pip3_path)
    else
        echo "Unable to locate pip3. Exiting script."
        exit 1
    fi
fi

# Install Python dependencies including torch, torchvision, and torchaudio
pip3 install torch==2.4.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install matplotlib pandas==2.0.0

# Configure SSH server (only for native Ubuntu, not needed in WSL)
if [ "$IS_WSL" = false ]; then
    sudo mkdir -p /var/run/sshd
    echo 'root:root' | sudo chpasswd
    sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

    # Start SSH service
    sudo systemctl enable ssh
    sudo systemctl start ssh
fi

echo "Setup complete."
