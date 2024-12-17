#!/bin/bash

# Exit immediately if a command fails
set -e

# Variables
service_name="candy"  # Service name from docker-compose.yml
new_image_name="intellistream/llh"  # Docker Hub repository name
tag="devel-ubuntu22.04"  # Image tag

# Get the container ID dynamically using the service name
echo "Retrieving container ID for service: $service_name..."
container_id=$(docker-compose ps -q "$service_name")

# Check if the container exists
if [ -z "$container_id" ]; then
    echo "Error: No container found for service '$service_name'. Is it running?"
    exit 1
fi

echo "Found container ID: $container_id"

# Commit the container to a new image
new_image="${new_image_name}:${tag}"
echo "Committing container '$container_id' to image '$new_image'..."
docker commit "$container_id" "$new_image"

# Verify the new image
echo "Image committed successfully. Listing the new Docker image:"
docker images | grep "$new_image_name"

# Push the image to Docker Hub
echo "Pushing image '$new_image' to Docker Hub..."
docker push "$new_image"

# Verify the push was successful
if [ $? -eq 0 ]; then
    echo "Image '$new_image' pushed to Docker Hub successfully!"
else
    echo "Error: Failed to push image '$new_image' to Docker Hub."
    exit 1
fi
