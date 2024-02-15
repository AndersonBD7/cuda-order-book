cmake_minimum_required(VERSION 3.18)
project(cuda-order-book LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# CUDA Architecture: Target Ampere (A100/RTX30xx) and Turing (T4)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Compiler Flags for Performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas -v")

# Include Directories
include_directories(include)

# Find CUDA Package
find_package(CUDAToolkit REQUIRED)

# Source Files
file(GLOB_RECURSE SOURCES 
    "src/kernels/*.cu"
    "src/host/*.cpp"
)

# Executable
add_executable(lob_engine ${SOURCES})

# Libraries
target_link_libraries(lob_engine PRIVATE CUDA::cudart)

# Install Script
install(TARGETS lob_engine DESTINATION bin)