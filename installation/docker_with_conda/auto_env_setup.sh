#!/bin/bash

# Check if Hugging Face token is provided as an argument
if [ -z "$1" ]; then
  echo "Error: Please provide your Hugging Face token as an argument."
  echo "Usage: bash auto_env_setup.sh <HUGGING_FACE_TOKEN>"
  exit 1
fi

# Source Conda to activate environment properly
source /opt/conda/bin/activate
conda activate flow 

# Install Hugging Face dependencies
pip install torch==2.4.0 huggingface_hub

# Log in to Hugging Face using token from the first argument
huggingface-cli login --token "$1"

# Update environment and install specific dependencies
conda env update --name flow --file environment.yml
