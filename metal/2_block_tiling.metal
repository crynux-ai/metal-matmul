#include <metal_stdlib>
#include "param.metal"

using namespace metal;

template<ushort BLOCK_K, ushort BLOCK_N>
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
    float val[BLOCK_N][BLOCK_N] = {0.};
    uint tx = thread_idx.x * BLOCK_N;
    uint ty = thread_idx.y * BLOCK_N;

{
    threadgroup float* a_cache = shmem;
    threadgroup float* b_cache = shmem + tdim.x * BLOCK_N * BLOCK_K;
    uint tigx_bk = tpitg.x * BLOCK_N * BLOCK_K;
    uint tigy_bk = tpitg.y * BLOCK_N * BLOCK_K;
    uint tx_k = tx * param->K;
    uint ty_k = ty * param->K;

    float a_value[BLOCK_N];
    float b_value[BLOCK_N];

    for (uint i = 0; i < param->K; i+=BLOCK_K) {
        for (uint jy = tpitg.y, jx = tpitg.x; jy < BLOCK_N * BLOCK_K; jy+= tdim.y, jx+=tdim.x) {
            a_cache[tigx_bk + jy] = A[tx_k + jy / BLOCK_K * param->K + i + jy % BLOCK_K];
            b_cache[tigy_bk + jx] = B[ty_k + jx / BLOCK_K * param->K + i + jx % BLOCK_K];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_K; k++) {
            uint ac_ptr = tigx_bk + k;
            uint bc_ptr = tigy_bk + k; 
            for (uint x = 0; x < BLOCK_N; x++, ac_ptr+=BLOCK_K, bc_ptr+=BLOCK_K) {
                a_value[x] = a_cache[ac_ptr];
                b_value[x] = b_cache[bc_ptr];
            }
            for (uint x = 0; x < BLOCK_N; x++) {
                for (uint y = 0; y < BLOCK_N; y++) {
                    val[x][y] += a_value[x] * b_value[y];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

    uint tx_m = tx * param->M + ty;
    for (uint x = 0; x < BLOCK_N; x++) {
        uint ptr = tx_m + x * param->M;
        for (uint y = 0; y < BLOCK_N; y++, ptr++) {
            output[ptr] = val[x][y];
        }
    }
}


typedef decltype(block_tiling<1, 1>) block_tiling_fn;

template [[host_name("block_tiling_1_4")]] kernel block_tiling_fn block_tiling<1, 4>;
template [[host_name("block_tiling_1_8")]] kernel block_tiling_fn block_tiling<1, 8>;
template [[host_name("block_tiling_1_16")]] kernel block_tiling_fn block_tiling<1, 16>;
template [[host_name("block_tiling_2_4")]] kernel block_tiling_fn block_tiling<2, 4>;
template [[host_name("block_tiling_2_8")]] kernel block_tiling_fn block_tiling<2, 8>;
template [[host_name("block_tiling_2_16")]] kernel block_tiling_fn block_tiling<2, 16>;
template [[host_name("block_tiling_4_2")]] kernel block_tiling_fn block_tiling<4, 2>;
template [[host_name("block_tiling_4_4")]] kernel block_tiling_fn block_tiling<4, 4>;
template [[host_name("block_tiling_4_8")]] kernel block_tiling_fn block_tiling<4, 8>;
template [[host_name("block_tiling_4_16")]] kernel block_tiling_fn block_tiling<4, 16>;
template [[host_name("block_tiling_8_2")]] kernel block_tiling_fn block_tiling<8, 2>;
template [[host_name("block_tiling_8_4")]] kernel block_tiling_fn block_tiling<8, 4>;
template [[host_name("block_tiling_8_8")]] kernel block_tiling_fn block_tiling<8, 8>;
template [[host_name("block_tiling_8_16")]] kernel block_tiling_fn block_tiling<8, 16>;

