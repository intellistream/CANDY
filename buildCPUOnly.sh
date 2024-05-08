#!/bin/bash

echo "First, make sure you have sudo"
sudo ls
echo "Installing others..."
sudo apt install -y liblapack-dev libblas-dev
sudo apt-get install -y graphviz
sudo apt-get install -y libcudnn8 libcudnn8-dev
pip install matplotlib pandas==2.0.0
pip install torch==1.13.0 --index-url https://download.pytorch.org/whl/cpu
echo "Build CANDY and PyCandy"
# Step 1: Configure the project
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build &&cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DENABLE_HDF5=ON -DENABLE_PYBIND=ON -DCMAKE_INSTALL_PREFIX=/usr/local/lib ..

# Step 2: Determine the maximum number of threads
max_threads=$(nproc)

# Step 3: Build the project using the maximum number of threads
cmake --build . --parallel $max_threads
sudo cmake --install .
