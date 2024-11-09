# Source Conda to activate environment properly
source /opt/conda/bin/activate
conda activate flow 

# Install Hugging Face and login
pip install torch==2.4.0 huggingface_hub
# Please config your own hugging face token.
huggingface-cli login --token "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" 

# Update environment and install specific dependencies
conda env update --name flow --file environment.yml