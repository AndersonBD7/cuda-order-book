#include "../../include/core/types.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace lob {
namespace kernels {

    // Warp-level reduction for finding the best price efficiently
    // This avoids shared memory bank conflicts
    __device__ __forceinline__ Price warp_reduce_max(Price val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            Price other = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val = (val > other) ? val : other;
        }
        return val;
    }

    __device__ __forceinline__ Price warp_reduce_min(Price val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            Price other = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val = (val < other) ? val : other;
        }
        return val;
    }

    /*
     * Kernel: match_orders_batch
     * Description: Matches incoming market orders against a limit order book
     * stored in global memory. Uses atomic operations for thread safety.
     */
    __global__ void match_orders_batch(
        const Order* __restrict__ incoming_orders,
        int num_orders,
        Order* __restrict__ book_orders,
        int* book_depth,
        Trade* trades,
        int* trade_count
    ) {
        // Global thread index
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_orders) return;

        // Load order into register (L1 Cache)
        Order my_order = incoming_orders[tid];
        
        // Simplified Logic: Direct matching against opposite side
        // In production, we would use a red-black tree flattened array
        
        if (my_order.type == OrderType::LIMIT) {
            // Atomic insertion into the book
            int idx = atomicAdd(book_depth, 1);
            if (idx < MAX_ORDERS) {
                book_orders[idx] = my_order;
            }
        }
        else if (my_order.type == OrderType::MARKET) {
            // Market Order Matching Logic
            // Scan the book for best price
            // Note: This is a naive O(N) scan for demonstration. 
            // Real implementation uses a parallel bitonic sort.
            
            for (int i = 0; i < *book_depth; ++i) {
                Order existing = book_orders[i];
                
                // Check if sides are opposite
                if (existing.quantity > 0 && existing.side != my_order.side) {
                    bool price_match = false;
                    
                    if (my_order.side == Side::BUY && existing.price <= my_order.price) {
                        price_match = true;
                    } else if (my_order.side == Side::SELL && existing.price >= my_order.price) {
                        price_match = true;
                    }

                    if (price_match) {
                        // Execute Trade using Atomic Compare-And-Swap (CAS) to prevent race conditions
                        Quantity trade_qty = min(my_order.quantity, existing.quantity);
                        
                        // Decrement quantity in book safely
                        Quantity old_qty = atomicSub(&book_orders[i].quantity, trade_qty);
                        
                        if (old_qty >= trade_qty) {
                            // Successful trade capture
                            int trade_idx = atomicAdd(trade_count, 1);
                            if (trade_idx < MAX_TRADES) {
                                trades[trade_idx].buy_id = (my_order.side == Side::BUY) ? my_order.id : existing.id;
                                trades[trade_idx].sell_id = (my_order.side == Side::SELL) ? my_order.id : existing.id;
                                trades[trade_idx].price = existing.price;
                                trades[trade_idx].quantity = trade_qty;
                                trades[trade_idx].timestamp = clock64(); // GPU Hardware Clock
                            }
                            my_order.quantity -= trade_qty;
                        } else {
                            // Trade failed due to race condition, restore quantity
                            atomicAdd(&book_orders[i].quantity, trade_qty);
                        }
                    }
                }
                if (my_order.quantity == 0) break;
            }
        }
    }
} // namespace kernels
} // namespace lob