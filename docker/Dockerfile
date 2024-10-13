# Base image with CUDA 11.7 and Python 3.10
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install essential packages and dependencies
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    sudo \
    build-essential \
    cmake \
    curl \
    unzip \
    liblapack-dev \
    libblas-dev \
    libboost-dev \
    graphviz \
    python3 \
    python3-pip \
    openssh-server \
    gdb \
    liblog4cxx-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install matplotlib pandas==2.0.0

# Set up CUDA environment variables
ENV CUDACXX=/usr/local/cuda/bin/nvcc

# Configure SSH
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Set up workspace directory
WORKDIR /workspace

# Default command to run SSH server
CMD ["/usr/sbin/sshd", "-D"]
