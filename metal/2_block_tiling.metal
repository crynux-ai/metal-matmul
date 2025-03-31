#include <metal_stdlib>
#include "param.metal"

using namespace metal;

kernel void block_tiling(
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
    uint BLOCK_K = tdim.y;
    threadgroup float* a_cache = shmem;
    threadgroup float* b_cache = shmem + tdim.x * param->BLOCK_N * BLOCK_K;
    float val[32][32] = {0.};

    uint tig_x = tpitg.x * param->BLOCK_N;
    uint tig_y = tpitg.y * param->BLOCK_M;
    uint tx = thread_idx.x * param->BLOCK_N;
    uint ty = thread_idx.y * param->BLOCK_M;

    uint istep = param->K / BLOCK_K;

    for (uint i = 0; i < istep; i++) {
        // Assume tdim.x == tdim.y == BLOCK_K
        for (uint j = 0; j < param->BLOCK_N; j++) {
            a_cache[(tig_x + j) * BLOCK_K + tpitg.y] = A[(tx + j) * param->K + i * tdim.y + tpitg.y];
            b_cache[(tig_y + j) * BLOCK_K + tpitg.x] = B[(ty + j) * param->K + i * tdim.x + tpitg.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);


        for (uint x = 0; x < param->BLOCK_N; x++) {
            for (uint y = 0; y < param->BLOCK_M; y++) {
                for (uint k = 0; k < BLOCK_K; k++) {
                    uint ac_ptr = (tig_x + x) * BLOCK_K + k;
                    uint bc_ptr = (tig_y + y) * BLOCK_K + k;
                    val[x][y] += a_cache[ac_ptr] * b_cache[bc_ptr];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint x = 0; x < param->BLOCK_N; x++) {
        for (uint y = 0; y < param->BLOCK_M; y++) {
            uint ptr = (thread_idx.x * param->BLOCK_N + x) * param->M + thread_idx.y * param->BLOCK_M + y;
            output[ptr] = val[x][y];
        }
    }
}
