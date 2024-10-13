# Base image with CUDA 11.7 and Python 3.10
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install essential packages and dependencies
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN apt-get update -q && apt-get install -y -q --no-install-recommends \
    --allow-downgrades \
    --allow-remove-essential \
    --allow-change-held-packages \
    sudo \
    build-essential \
    cmake \
    curl \
    wget \
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
    cuda-command-line-tools-11-7 \
    libcudnn8 \
    libcudnn8-dev \
    locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

# Install Libtorch compatible with CUDA 11.6 or earlier
# Download libtorch.zip separately
RUN wget -q --retry-connrefused --tries=5 --waitretry=5 -O libtorch.zip https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-1.13.1%2Bcu117.zip

# Extract Libtorch if the download was successful
RUN if [ -f libtorch.zip ]; then \
       unzip libtorch.zip -d /usr/local/ && \
       rm libtorch.zip; \
    else \
       echo "Libtorch download failed."; \
       exit 1; \
    fi && \
    if [ ! -d "/usr/local/libtorch" ]; then \
       echo "Extraction of Libtorch failed"; \
       exit 1; \
    fi

# Set environment variables for CUDA and cuDNN
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDNN_INCLUDE_DIR=/usr/include
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu
ENV CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so
ENV CMAKE_PREFIX_PATH="/usr/local/libtorch/share/cmake/Torch"
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"

# Configure SSH (For Development Only, avoid in production)
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords no/' /etc/ssh/sshd_config

# Set up workspace directory
WORKDIR /workspace

# Default command to run SSH server
CMD ["/usr/sbin/sshd", "-D"]