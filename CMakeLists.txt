cmake_minimum_required(VERSION 3.22)
project(cuda-order-book VERSION 2.0.0 LANGUAGES CXX CUDA)

# --- Options ---
option(LOB_BUILD_TESTS "Build Unit Tests" ON)
option(LOB_BUILD_BENCHMARKS "Build Benchmarks" ON)
option(LOB_ENABLE_PROFILING "Enable NVTX Profiling" OFF)

# --- CUDA Architecture Setup ---
# Auto-detect or target high-end GPUs (A100, H100)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90") 
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas -v --expt-relaxed-constexpr")

# --- Include Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# --- Core Library ---
add_library(lob_core SHARED
    src/core/order_book.cpp
    src/core/matching_engine.cpp
    src/device/kernels/match_kernel.cu
    src/device/kernels/sort_kernel.cu
)

target_link_libraries(lob_core PUBLIC CUDA::cudart)

# --- Benchmarks ---
if(LOB_BUILD_BENCHMARKS)
    add_executable(bench_throughput benchmarks/throughput/main_bench.cu)
    target_link_libraries(bench_throughput PRIVATE lob_core)
endif()

# --- Tests ---
if(LOB_BUILD_TESTS)
    enable_testing()
    # GTest integration placeholder
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")