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
|naive 8x8        |     79      |      138      |       135     |      141      |  |
|global mem       |    102      |       93      |        90     |       79      |  |
|tg mem 16x16     |     81      |      397      |       433     |      434      |  |
|tg mem unroll    |     96      |      786      |       900     |      906      | unroll 4 |
|block tiling     |     35      |      227      |       253     |      217      | thread group 16x16, block_k 16, block_n/m 4 |


## Naive

Different thread group size.

|Method           | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|-----------------|-------------|---------------|---------------|---------------|--------|
|naive 1x1        |     45      |       38      |        26     |       13      |  |
|naive 4x4        |     70      |      112      |       109     |      106      |  |
|naive 8x8        |     79      |      138      |       135     |      141      |  |
|naive 16x16      |     81      |       93      |        90     |       80      |  |

## Threadgroup memory

Load data from memory to threadgroup
Thread group memory == 32768

|Method           | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|-----------------|-------------|---------------|---------------|---------------|--------|
|tg mem 1x1       |    102      |       93      |        90     |       79      |  |
|tg mem 8x8       |     41      |       70      |        69     |       69      |  |
|tg mem 16x16     |     61      |      247      |       252     |      251      |  |
|tg mem 32x32     |     47      |       87      |        90     |       90      |  |
|tg mem 16x16 unroll 4   |     81      |       479      |        495     |       471      |  |
|tg mem 16x16 unroll 8   |     81      |       528      |        522     |       528      |  |
|tg mem 16x16 unroll 16  |     87      |       476      |        480     |       475      |  |
|tg mem 16x16 unroll 32  |     67      |       237      |        245     |       245      |  |

Thread group memory == tdim.x x tdim.y x 8

|Method           | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|-----------------|-------------|---------------|---------------|---------------|--------|
|tg mem 4x4       |     60      |      211      |       220     |      169      |  |
|tg mem 8x8       |     80      |      432      |       466     |      367      |  |
|tg mem 16x16     |     81      |      397      |       433     |      434      |  |
|tg mem 32x32     |     45      |       96      |        98     |       99      |  |
|tg mem 16x16 unroll 2   |     86      |       678      |        755     |       726      |  |
|tg mem 16x16 unroll 4   |     99      |       810      |        900     |       903      |  |
|tg mem 16x16 unroll 8   |     93      |       791      |        866     |       869      |  |
|tg mem 16x16 unroll 16  |     93      |       703      |        758     |       755      |  |



## Acknowledge

* https://github.com/siboehm/SGEMM_CUDA
* https://github.com/bkvogel/metal_performance_testing
* https://developer.apple.com/forums/thread/105534
