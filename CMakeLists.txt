cmake_minimum_required(VERSION 3.16)
project(GenerateWeights CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with full optimizations
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -march=native -funroll-loops)

# Find pthreads (needed for std::thread on Linux)
find_package(Threads REQUIRED)

add_executable(GenerateWeights GenerateWeights.cpp)

target_link_libraries(GenerateWeights
    PRIVATE
    Threads::Threads
    stdc++fs   # std::filesystem on older GCC
)

# Print build summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")