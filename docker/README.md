1. Replace `id_rsa.pub` to your own public key.
2. Start the container with Docker Compose: `docker-compose up -d`
2. Open CLion and navigate to `File > Settings > Build, Execution, Deployment > Toolchains`.
3. Add a Remote Host toolchain:
   - Host: `localhost`
   - Port: `2222`
   - Username: `developer`
   - Password: `password` (or configure with SSH key)
4. For compilers and CMake, set:
   - C Compiler: `/usr/bin/gcc`
   - C++ Compiler: `/usr/bin/g++`
   - CMake: `/usr/bin/cmake`
5. Configure CMake Profile:
   - Go to `File > Settings > Build, Execution, Deployment > CMake`.
