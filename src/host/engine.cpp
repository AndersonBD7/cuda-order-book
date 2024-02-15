#include "../../include/core/types.cuh"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

// Forward declaration of the kernel wrapper
namespace lob {
namespace kernels {
    __global__ void match_orders_batch(const Order*, int, Order*, int*, Trade*, int*);
}
}

namespace lob {

class MatchingEngine {
private:
    Order* d_incoming_orders;
    Order* d_book_orders;
    Trade* d_trades;
    int* d_book_depth;
    int* d_trade_count;
    
    cudaStream_t stream;

public:
    MatchingEngine() {
        // Allocate GPU memory
        cudaMalloc(&d_incoming_orders, MAX_ORDERS * sizeof(Order));
        cudaMalloc(&d_book_orders, MAX_ORDERS * sizeof(Order));
        cudaMalloc(&d_trades, MAX_TRADES * sizeof(Trade));
        cudaMalloc(&d_book_depth, sizeof(int));
        cudaMalloc(&d_trade_count, sizeof(int));

        cudaMemset(d_book_depth, 0, sizeof(int));
        cudaMemset(d_trade_count, 0, sizeof(int));
        
        cudaStreamCreate(&stream);
    }

    ~MatchingEngine() {
        cudaFree(d_incoming_orders);
        cudaFree(d_book_orders);
        cudaFree(d_trades);
        cudaFree(d_book_depth);
        cudaFree(d_trade_count);
        cudaStreamDestroy(stream);
    }

    void process_batch(const std::vector<Order>& host_orders) {
        if (host_orders.size() > MAX_ORDERS) {
            throw std::runtime_error("Batch size exceeds GPU buffer limit");
        }

        // 1. Async Copy: Host -> Device
        cudaMemcpyAsync(d_incoming_orders, host_orders.data(), 
                        host_orders.size() * sizeof(Order), 
                        cudaMemcpyHostToDevice, stream);

        // 2. Launch Kernel
        int threads = 256;
        int blocks = (host_orders.size() + threads - 1) / threads;

        lob::kernels::match_orders_batch<<<blocks, threads, 0, stream>>>(
            d_incoming_orders,
            host_orders.size(),
            d_book_orders,
            d_book_depth,
            d_trades,
            d_trade_count
        );

        // 3. Error Check
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(err));
        }

        // 4. Synchronize (or use events for overlapping)
        cudaStreamSynchronize(stream);
    }

    int get_trade_count() {
        int count;
        cudaMemcpy(&count, d_trade_count, sizeof(int), cudaMemcpyDeviceToHost);
        return count;
    }
};

} // namespace lob

int main() {
    try {
        lob::MatchingEngine engine;
        std::cout << "[GPU Engine] Initialized successfully." << std::endl;
        std::cout << "[GPU Engine] Warming up CUDA context..." << std::endl;
        
        // Simulation data would go here
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}