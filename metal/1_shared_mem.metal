#include <metal_stdlib>
#include "param.metal"

using namespace metal;

kernel void shared_mem(
    constant RunParams *param  [[ buffer(0) ]],
    const device float *A      [[ buffer(1) ]],     // [N, K]
    const device float *B      [[ buffer(2) ]],     // [M, K]
    device float *output       [[ buffer(3) ]],     // [N, M]
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint2 thread_idx           [[ thread_position_in_grid ]],
    uint2 tg_dim               [[ threadgroups_per_grid ]],
    uint2 tg_idx               [[threadgroup_position_in_grid]],
    ushort2 tpitg              [[thread_position_in_threadgroup]],
    ushort2 tdim               [[threads_per_threadgroup]]
) {
    uint a_ptr = thread_idx.x * param->K;
    uint b_ptr = thread_idx.y * param->K;
    threadgroup float* a_cache = shmem;
    threadgroup float* b_cache = shmem + tdim.x * tdim.y;
    float val = 0;
    
    for (uint i = 0; i < param->K; i+=tdim.y, a_ptr+=tdim.y, b_ptr+=tdim.x) {
        // Assume tdim.x == tdim.y == tile_width
        a_cache[tpitg.x * tdim.y + tpitg.y] = A[a_ptr + tpitg.y];
        b_cache[tpitg.y * tdim.x + tpitg.x] = B[b_ptr + tpitg.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ac_ptr = tpitg.x * tdim.y;
        uint bc_ptr = tpitg.y * tdim.x;
        for (uint z = 0; z < tdim.y; z++, ac_ptr++, bc_ptr++) {
            val += a_cache[ac_ptr] * b_cache[bc_ptr];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint ptr = thread_idx.x * param->M + thread_idx.y;
    output[ptr] = val;
}
