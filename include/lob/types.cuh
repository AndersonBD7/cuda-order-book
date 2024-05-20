#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lob {
    
    using OrderId = uint64_t;
    using Price = uint64_t;
    using Qty = uint32_t;

    // 16-byte aligned for coalesced memory access
    struct __align__(16) Order {
        OrderId id;
        Price price;
        Qty qty;
        uint32_t type; // 0: Limit, 1: Market
        uint64_t timestamp;
    };

    struct Trade {
        OrderId buy_id;
        OrderId sell_id;
        Price price;
        Qty qty;
        uint64_t timestamp;
    };
}
