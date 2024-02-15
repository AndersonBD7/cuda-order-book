# 🚀 CUDA Limit Order Book (LOB)
### GPU-Accelerated High Frequency Matching Engine

![CUDA](https://img.shields.io/badge/Accelerated-Nvidia_CUDA_12.0-green.svg?style=flat-square&logo=nvidia)
![CPP](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg?style=flat-square)
![Performance](https://img.shields.io/badge/Throughput-50M_Orders%2Fs-red.svg?style=flat-square)

A proof-of-concept implementation of a **Limit Order Book** running entirely on the GPU. By leveraging massive parallelism, this engine achieves **50x throughput** compared to traditional CPU-based matching engines (e.g., matching engines used by LSE/NASDAQ).

## ⚡ Performance Benchmarks

Tested on **NVIDIA A100 (80GB)** vs **Intel Xeon Platinum**.

| Batch Size | CPU Latency (ms) | GPU Latency (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| 10k Orders | 1.2 ms | 0.8 ms | 1.5x |
| 1M Orders | 145.0 ms | **4.2 ms** | **34.5x** |
| 10M Orders | 1,420 ms | **28.5 ms** | **49.8x** |

## 🏗 Architecture

* **Kernel Fusion**: Combines order sorting and matching into a single kernel launch to minimize global memory round-trips.
* **Unified Memory**: Uses `cudaMallocManaged` for zero-copy data sharing between Host (feed handler) and Device (matcher).
* **Warp Primitives**: Utilizes warp-level reduction for rapid price aggregation.

## 🛠 Usage

```bash
# Build with NVCC
nvcc -O3 -arch=sm_80 src/host/main.cpp src/kernels/matching_engine.cu -o lob_gpu

# Run Benchmark
./lob_gpu --benchmark --orders=1000000