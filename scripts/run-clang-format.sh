#!/bin/bash

# This script will format all C/C++ files in the repository using clang-format.
# The .clang-format file should be placed at the root of the repository.

set -e

# Define the list of file types to format.
file_types=("*.h" "*.hpp" "*.c" "*.cpp")

# Find and format all files with the specified file types.
for file_type in "${file_types[@]}"
do
  find . -name "$file_type" -exec clang-format -i {} +
done
