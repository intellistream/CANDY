#!/bin/bash

# Source Conda to activate environment properly
source /opt/conda/bin/activate
conda activate flow

# Install Hugging Face dependencies
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch==2.4.0 huggingface_hub

# Update environment and install specific dependencies
conda env update --name flow --file environment.yml
