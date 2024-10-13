#!/bin/bash

# Build and run the Docker container
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Display SSH connection information
echo "Docker container is running. You can connect via SSH with:"
echo "ssh root@<remote_server_ip> -p 2222"