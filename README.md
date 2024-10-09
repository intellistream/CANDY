# ModernCPlusProjectTemplate [![CMake](https://github.com/intellistream/ModernCPlusProjectTemplate/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/intellistream/ModernCPlusProjectTemplate/actions/workflows/cmake.yml)

Template for C++ 20 project; using CMAKE to manage

## Requires G++11
```shell
sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu hirsute main universe'
sudo apt-get update"
sudo apt install gcc-11 g++-11"
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
```

## Requires Log4cxx
```shell
sudo apt-get install -y liblog4cxx-dev
```

## Code Structure
- benchmark -- application code to use the generated shared library
- cmake -- cmake configuration files
- docs -- any documents
- include -- all the header files
- src -- corresponding source files, will generate a shared library
- test -- test code based on google test

## How to start?
- Please rename IntelliStream to your desired project name.
- Implement your core function code in src and include folders.
- Implement your application code in benchmark folder.
- Implement your test code in test folder.
