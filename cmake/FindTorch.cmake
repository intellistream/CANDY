# Run the Python command and capture its output
execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PYTHON_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Debug message to print the output from Python
message(STATUS "PyTorch CMake path from Python: ${PYTHON_OUTPUT}")

# Check if PYTHON_OUTPUT is valid
if(NOT PYTHON_OUTPUT)
    message(FATAL_ERROR "Could not find PyTorch CMake path. Make sure PyTorch is installed.")
endif()

# Append the Python command output to CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYTHON_OUTPUT}")

# Debug message to print the updated CMAKE_PREFIX_PATH
message(STATUS "Updated CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
