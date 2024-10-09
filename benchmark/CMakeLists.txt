# Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (BUILD_BENCHMARKS)
    message("---Building benchmarks ${source_dir}")
    include_directories("${source_dir}/include/")
    include_directories("include/")
    add_executable(benchmark "src/Benchmark.cpp")
    target_link_libraries(benchmark IntelliStream)

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/datasets
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    message("---Done building benchmarks")
else (BUILD_BENCHMARKS)
    message("---Not building benchmarks!")
endif (BUILD_BENCHMARKS)