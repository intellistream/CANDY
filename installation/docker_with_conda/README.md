## Docker container setup with conda environment.

This docker container setup is extended from `../docker/`. All docker related settings and prerequisites is compatible if you only want to setup `conda` env.

This folder also include a one-click conda env setup script `auto_env_setup.sh` for `FlowRAG` with `Haystack` dependencies.
The default workspace folder `/workspace` contains two loaded files `environment.yml` and `auto_env_setup.sh`. 
This script can be run after container is running with the following steps:

1. Enter container `/bin/bash` CLI via Docker CLI `docker exec -it $container_name /bin/bash` or using SSH `ssh -p 2222 root@$CONTAINER_IP`.
2. Before start, make sure to configure your `huggingface token` that can download associated model from HuggingFace.
3. After configuration, run `bash auto_env_setup.sh` to setup conda env `flow`, which can be used in PyCharm for FlowRAG.
