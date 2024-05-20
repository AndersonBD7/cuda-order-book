#pragma once
#include <cuda_runtime.h>

namespace lob {
namespace device {

    // Warp-level primitive using PTX for faster atomic increment
    __device__ __forceinline__ uint32_t atomicAggInc(uint32_t* ctr) {
        uint32_t active = __activemask();
        uint32_t leader = __ffs(active) - 1;
        uint32_t lane = threadIdx.x % 32;
        uint32_t res;
        
        if (lane == leader) {
            res = atomicAdd(ctr, __popc(active));
        }
        res = __shfl_sync(active, res, leader);
        return res + __popc(active & ((1 << lane) - 1));
    }
}
}
