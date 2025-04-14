#include <metal_stdlib>
#include "param.metal"

using namespace metal;

kernel void float4x4_kernel(
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
    float4x4 val(0.);
    uint tx = thread_idx.x * 4;
    uint ty = thread_idx.y * 4;

    threadgroup float* a_cache = shmem;
    threadgroup float* b_cache = shmem + tdim.x * 4 * 4;
    uint tigx_bk = tpitg.x * 4 * 4;
    uint tigy_bk = tpitg.y * 4 * 4;
    uint tx_k = tx * param->K;
    uint ty_k = ty * param->K;
    float4x4 a_value;
    float4x4 b_value;

    for (uint i = 0; i < param->K; i+=4) {
        a_cache[tigx_bk + tpitg.y] = A[tx_k + tpitg.y / 4 * param->K + i + tpitg.y % 4];
        b_cache[tigy_bk + tpitg.x] = B[ty_k + tpitg.x / 4 * param->K + i + tpitg.x % 4];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(4)
        for (int x = 0; x < 4; x++) {
            #pragma unroll(4)
            for (int y = 0; y < 4; y++) {
                a_value[x][y] = a_cache[tigx_bk + x * 4 + y];
                b_value[y][x] = b_cache[tigy_bk + x * 4 + y];
            }   
        }

        val += b_value * a_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint tx_m = tx * param->M + ty;
    for (uint x = 0; x < 4; x++) {
        uint ptr = tx_m + x * param->M;
        for (uint y = 0; y < 4; y++, ptr++) {
            output[ptr] = val[x][y];
        }
    }
}

