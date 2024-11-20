# Docker Container Setup with Conda Environment for FlowRAG

This Docker container is pre-configured to set up a Conda environment specifically for the `FlowRAG` project, including `Haystack` dependencies. It builds on the Docker settings from `../docker/` and offers a streamlined one-click environment setup.

## Setting up the Docker Container

To initialize this container, simply run the provided setup script:

```bash
bash start.sh
```

This script will build and start the Docker container with all necessary configurations.

## Conda Environment Setup with `auto_env_setup.sh`

Once the Docker container is running, you can configure the Conda environment using `auto_env_setup.sh`. This script sets up the `flow` environment with all required dependencies. 

### Files Included in `/workspace`

- `environment.yml`: Defines the Conda environment configuration for FlowRAG.
- `auto_env_setup.sh`: Automates the Conda environment setup inside the container.

### Steps to Run `auto_env_setup.sh`

1. **Access the Running Container**: Connect to the container’s CLI either through Docker or SSH:
   - Docker CLI:
     ```bash
     docker exec -it <container_name> /bin/bash
     ```
   - SSH (if configured):
     ```bash
     ssh -p 2222 root@<CONTAINER_IP>
     ```

2. **Apply Hugging Face Token**: Ensure that you have your Hugging Face token to enable downloading models from Hugging Face.

3. **Run the Setup Script**: Once connected, execute the script to set up the Conda environment:
   ```bash
   bash auto_env_setup.sh $your_token_here
   ```

The environment `flow` will then be ready for use, and you can configure it in PyCharm or other IDEs to start working with `flow`.
