# Base image with CUDA, Python, and required development tools
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install essential packages and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    openssh-server \
    build-essential \
    cmake \
    git \
    curl \
    unzip \
    liblapack-dev \
    libblas-dev \
    libboost-dev \
    graphviz \
    python3 \
    python3-pip \
    python3-dev \
    liblog4cxx-dev \
    rsync \
    gdb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install matplotlib pandas==2.0.0

# Set up CUDA environment variables
ENV CUDACXX=/usr/local/cuda/bin/nvcc

# Set up SSH for remote access
RUN mkdir /var/run/sshd && \
    useradd -m developer && \
    echo "developer:password" | chpasswd && \
    mkdir -p /home/developer/.ssh && \
    chmod 700 /home/developer/.ssh

# Copy SSH key into the container (replace with your public key file path)
COPY id_rsa.pub /home/developer/.ssh/authorized_keys
RUN chmod 600 /home/developer/.ssh/authorized_keys && \
    chown -R developer:developer /home/developer/.ssh

# Expose SSH port for remote connection
EXPOSE 22

# Switch shell to bash
SHELL ["/bin/bash", "-c"]

# Default command to run SSHD when container starts
CMD ["/usr/sbin/sshd", "-D"]