# Project Setup Instructions

This guide will help you set up the environment on **Windows** using **WSL (Windows Subsystem for Linux)** with **Ubuntu 22.04**, and build the project using CMake.

## Prerequisites

1. **Windows 10 or 11** with **WSL** enabled.
2. **Ubuntu 22.04** installed in WSL.

## Installation Steps

### 1. Enable WSL on Windows

- Open **PowerShell** as Administrator and run the following command to enable WSL and set the default version to WSL 2:

  ```powershell
  wsl --install
  ```

- Restart your computer if prompted.

### 2. Install Ubuntu 22.04 in WSL

- Once WSL is enabled, install **Ubuntu 22.04** by running the following command in **PowerShell**:

  ```powershell
  wsl --install -d Ubuntu-22.04
  ```

- After installation, open Ubuntu from the Start menu, and it will ask you to create a user and password for your WSL environment.

### 3. Clone the Project

- Open **Ubuntu (WSL)** and clone the project repository:

  ```bash
  git clone <https://github.com/intellistream/CANDY>
  cd <CANDY>
  ```

### 4. Run the Installation Script

- Execute the provided installation script to install all dependencies and set up the environment:

  ```bash
  bash install.sh
  ```

  This will:
    - Install essential system packages.
    - Install CUDA toolkit and PyTorch based on your GPU availability.
    - Configure environment variables for CUDA and Python.

### 5. Build the Project

- After running the installation script, proceed with building the project using **CMake** and **make**:

  ```bash
  mkdir build
  cd build
  cmake ..
  make
  ```

The project should now be built and ready to use.