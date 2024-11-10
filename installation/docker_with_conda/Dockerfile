# Base image with CUDA 11.7 and Python 3.11
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Environment variables
ENV USER=candy
ENV PASSWD=candy

# 设置环境变量，防止apt-get进行交互式操作
ENV DEBIAN_FRONTEND=noninteractive

# 替换系统的sources.list文件为使用中国镜像
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse\n\
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse\n\
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" > /etc/apt/sources.list

# Install essential packages and dependencies
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    sudo \
    build-essential \
    cmake \
    curl \
    unzip \
    sudo \
    liblapack-dev \
    libblas-dev \
    libboost-dev \
    rsync \
    nano \
    libcudnn8 \
    libcudnn8-dev \
    graphviz \
    python3 \
    python3-pip \
    openssh-server \
    gdb \
    liblog4cxx-dev \
    git \
    libcupti-dev \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    clang-format \
    && apt-get clean 

# Install Python dependencies
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip3 install torch matplotlib pandas==2.0.0

# Set up a new user and configure password
RUN useradd -m ${USER} && echo "${USER}:${PASSWD}" | chpasswd && usermod -G sudo ${USER}
# Please use this folder for your clion project
RUN mkdir /home/${USER}/candy
RUN chown -R ${USER} /home/${USER}/candy

# Configure SSH
USER root
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Add Conda to the PATH
ENV PATH="/opt/conda/bin:$PATH"

# Set up a workspace directory, defaulted to root for debugging
WORKDIR /workspace

# Create and activate the FlowRAG environment
COPY environment.yml auto_env_setup.sh /workspace/

# Install Conda and create flow venv
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    conda create -n flow python=3.11

# Expose SSH port
EXPOSE 22

# Default command to run SSH server
CMD ["/usr/sbin/sshd", "-D"]
