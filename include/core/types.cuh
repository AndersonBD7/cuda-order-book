#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Memory alignment for coalesced access on GPU
#define ALIGN(x) __align__(x)

namespace lob {

    using OrderId = uint64_t;
    using Price = uint32_t;
    using Quantity = uint32_t;

    enum class Side : uint8_t {
        BUY = 0,
        SELL = 1
    };

    enum class OrderType : uint8_t {
        LIMIT = 0,
        MARKET = 1,
        CANCEL = 2
    };

    // GPU-optimized Order Structure (32-byte aligned)
    struct ALIGN(16) Order {
        OrderId id;
        Price price;
        Quantity quantity;
        Side side;
        OrderType type;
        uint64_t timestamp; // Nanoseconds
    };

    // Result structure for matched trades
    struct ALIGN(16) Trade {
        OrderId buy_id;
        OrderId sell_id;
        Price price;
        Quantity quantity;
        uint64_t timestamp;
    };

    // Constants for static allocation
    constexpr int MAX_ORDERS = 1024 * 1024; // 1M orders per batch
    constexpr int MAX_TRADES = 1024 * 1024;
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = 256;

} // namespace lob