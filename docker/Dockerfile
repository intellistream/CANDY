# Base image with CUDA 11.7 and Python 3.10
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install essential packages and dependencies
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANGUAGE=en_US.UTF-8

# Install essential packages and dependencies
RUN apt-get update -q && apt-get install -y -q --allow-change-held-packages \
    sudo \
    build-essential \
    cmake \
    curl \
    unzip \
    liblapack-dev \
    libblas-dev \
    libboost-dev \
    libboost-all-dev \
    swig \
    graphviz \
    python3 \
    python3-pip \
    openssh-server \
    gdb \
    liblog4cxx-dev \
    git \
    libcudnn8 \
    libcudnn8-dev \
    nvidia-cuda-toolkit \
    locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

# Install Libtorch (example of installation process)
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.1%2Bcu117.zip && \
    if [ -f libtorch.zip ]; then \
        unzip libtorch.zip -d /usr/local/ && \
        rm libtorch.zip; \
    else \
        echo "Libtorch download failed."; \
        exit 1; \
    fi

# Verify that Libtorch has been correctly installed (Debugging)
RUN if [ ! -d "/usr/local/libtorch" ]; then echo "Extraction of Libtorch failed"; exit 1; fi

# Set Libtorch environment variables
ENV CMAKE_PREFIX_PATH="/usr/local/libtorch/share/cmake/Torch"

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:$PATH
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
