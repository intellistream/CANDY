- Start the container with Docker Compose: `docker-compose up -d`
- Open CLion and navigate to `File > Settings > Build, Execution, Deployment > Toolchains`.
- Add a Remote Host toolchain:
   - Host: `<remote_server_ip>`
   - Port: `2222`
   - Username: `root`
   - Password: `root` (or configure with SSH key)
4. For compilers and CMake, set:
   - C Compiler: `/usr/bin/gcc`
   - C++ Compiler: `/usr/bin/g++`
   - CMake: `/usr/bin/cmake`
5. Configure CMake Profile:
   - Go to `File > Settings > Build, Execution, Deployment > CMake`.
