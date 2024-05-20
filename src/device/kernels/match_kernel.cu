#include "../../../include/lob/types.cuh"
#include "../../../include/lob/device/primitives.cuh"

namespace lob {
namespace kernels {

    __global__ void match_orders_kernel(
        const Order* __restrict__ buy_orders,
        const Order* __restrict__ sell_orders,
        Trade* trades,
        uint32_t* trade_count,
        int num_orders
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_orders) return;

        // Load into registers (L1 Cache)
        Order buy = buy_orders[tid];
        Order sell = sell_orders[tid];

        // Simple price-time priority check
        if (buy.price >= sell.price && buy.qty > 0 && sell.qty > 0) {
            
            // Critical Section: Atomic aggregation to reduce contention
            uint32_t idx = device::atomicAggInc(trade_count);
            
            trades[idx].buy_id = buy.id;
            trades[idx].sell_id = sell.id;
            trades[idx].price = sell.price;
            trades[idx].qty = min(buy.qty, sell.qty);
            trades[idx].timestamp = clock64();
        }
    }
}
}
