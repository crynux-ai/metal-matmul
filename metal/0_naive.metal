#include <metal_stdlib>
#include "param.metal"

using namespace metal;

kernel void naive(
    constant RunParams *param  [[ buffer(0) ]],
    const device float *A      [[ buffer(1) ]],     // [N, K]
    const device float *B      [[ buffer(2) ]],     // [M, K]
    device float *output       [[ buffer(3) ]],     // [N, M]
    uint2 gid                  [[ thread_position_in_grid ]])
{
    float val = 0;
    uint a_ptr = gid.x * param->K;
    uint b_ptr = gid.y * param->K;
    for (uint i = 0; i < param->K; i++, a_ptr++, b_ptr++) {
        val += A[a_ptr] * B[b_ptr];
    }
    uint ptr = gid.x * param->M + gid.y;
    output[ptr] = val;
}
