#!/bin/bash

# This script will format all C/C++ files in the repository using clang-format.
# The .clang-format file should be placed at the root of the repository.

set -e

cd tools

mkdir -p build && cd build

cmake ..

cd ..

cmake --build build --config Release --target run_format

cd ..

found_clang_format=false
clang_format_executable="clang-format"
# get clang-format binary
if [ "$(command -v $clang_format_executable)" ]; then
    found_clang_format=true
fi
# maybe there is clang-format link
# from 11 - 18
if [ "$found_clang_format" = false ]; then
    for number in {11..18}
    do
        if [ "$(command -v clang-format-$number)" ]; then
            clang_format_executable="clang-format-$number"
            found_clang_format=true
            break
        fi
    done
fi
if [ "$found_clang_format" = true ]; then
    echo "Found clang-format executable: $clang_format_executable"
    tools/build/run_format $clang_format_executable --source_dirs apps test src include python_bindings
else
    echo "clang-format not found. Please install clang-format or add it to your PATH."
    exit 1
fi

rm -rf tools/build

