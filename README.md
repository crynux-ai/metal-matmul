# Apple Metal Matrix Multiplication

This is step-by-step optimization and performance benchmark for matmul on metal.


## Instruction


```
mkdir build && cd build
rm -R air_files && rm matmul.metallib
cmake .. && make && ./benchmark
```


## Stats

Device: M1 Max, 64GB, OS 15.3.2    (10.4 TFlops)
Performance (GFlops)

|Method           | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|-----------------|-------------|---------------|---------------|---------------|--------|
|MPS (Swift)      |     20      |     1262      |      7441     |     7428      |  |
|MPS (C++)        |    110      |     3943      |      7582     |     7377      |  |
|naive            |     45      |       38      |        26     |       13      |  |
|naive 16x16      |     81      |       93      |        90     |       80      |  |
|global mem       |    102      |       93      |        90     |       79      |  |
|threadgroup mem  |     81      |      391      |       421     |      422      | thread group 16x16 |
|block tiling     |     35      |      227      |       253     |      217      | thread group 16x16, block_k 16, block_n/m 4 |
|tg mem unroll    |     96      |      786      |       900     |      906      | unroll 4 |

## Acknowledge

* https://github.com/siboehm/SGEMM_CUDA
* https://github.com/bkvogel/metal_performance_testing
* https://developer.apple.com/forums/thread/105534
