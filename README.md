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
|block tiling     |     91      |     1778      |      2533     |     2380      | thread group 16x16, block_k 4, block_n 4 |
|block tiling     |     91      |     1778      |      2533     |     2380      | |


## Naive

Different thread group size.

|Method           | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|-----------------|-------------|---------------|---------------|---------------|--------|
|naive 1x1        |     45      |       38      |        26     |       13      |  |
|naive 4x4        |     70      |      112      |       109     |      106      |  |
|naive 8x8        |     79      |      138      |       135     |      141      |  |
|naive 16x16      |     81      |       93      |        90     |       80      |  |


* Read `2*K` per thread, we only consider read from memory.
* Compute `2*K` per thread, we only consider matmul computation, i.e. the necessary computation without index operation.
* Compute / IO: `1`

## Threadgroup memory

By reading data to threadgroup memory, we can:
* One-time reading from memory to threadgroup
* Much faster reading from threadgroup to compute
* Since IO is much slower than compute, we should see improvements when Compute/IO increases.

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


When BK == T:
* Threadgroup mem require `2*T^2*sizeof(float)`
* Read `2*T*K` per threadgroup, read `2*K/T` per thread;
* Compute `2 * T^2 * K` per threadgroup, `2*K` per thread;
* Compute / IO: `T`
* Concurrent thread in a threadgroup: `TGMem / (2*T^2 * sizeof(float))`

When BK > T:
* Threadgroup mem require `2*T*BK * sizeof(float)`
* Read `2*T^2*K/BK*BK/T` per threadgroup, read `2*K/T` per thread;
* Compute `2*T^2 * K/BK*BK` per threadgroup, `2*K` per thread;
* Compute / IO == `T`
* Concurrent threads in a threadgroup: `TGMem / (2*T*BK *sizeof(float))`

Adding BK will:
* lower num of concurrent threads by requsting more threadgroup memory
* no foundamental improvement from Compute/IO.

## Block tiling


### Impl

|Method                    | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|--------------------------|-------------|---------------|---------------|---------------|--------|
|tiling BK=1, BN=16, 16x16 |       7     |        80     |       133     |       129     |  |
|tiling BK=2, BN=8, 16x16  |      67     |      1405     |      1677     |      1591     |  |
|tiling BK=4, BN=4, 16x16  |      94     |      1741     |      2497     |      2362     |  |
|tiling BK=4, BN=8, 16x16  |      67     |      1555     |      1937     |      1932     |  |
|tiling BK=4, BN=16, 16x16 |      14     |       236     |       648     |       582     |  |
|tiling BK=8, BN=2, 16x16  |      94     |      1377     |      1669     |      1635     |  |
|tiling BK=8, BN=4, 16x16  |      80     |      1459     |      1818     |      1775     |  |
|tiling BK=8, BN=8, 16x16  |      53     |      1144     |      1398     |      1371     |  |

|tiling BK=1, BN=8 , 8x8   |      69     |      1041     |      1252     |       626     |  |
|tiling BK=1, BN=16, 8x8   |       9     |        91     |       115     |      1630     |  |
|tiling BK=2, BN=4, 8x8    |      85     |      1082     |       717     |       344     |  |
|tiling BK=2, BN=8, 8x8    |      84     |      1518     |      2076     |       987     |  |
|tiling BK=4, BN=2, 8x8    |     106     |      1059     |      1143     |       520     |  |
|tiling BK=4, BN=4, 8x8    |      65     |      1784     |      2083     |       627     |  |
|tiling BK=4, BN=8, 8x8    |      73     |      1557     |      2334     |      1211     |  |
|tiling BK=8, BN=2, 8x8    |      96     |      1504     |      1726     |       546     |  |
|tiling BK=8, BN=4, 8x8    |      88     |      1640     |      2169     |       702     |  |
|tiling BK=8, BN=8, 8x8    |      61     |      1119     |      1602     |      1387     |  |



* Threadgroup mem require `2*T*BN*BK*sizeof(float)`
* Read `2*T^2*K/BK*BK*BN/T` per threadgroup, read `2*K*BN/T` per thread;
* Compute `2*T^2*K*BN^2` per threadgroup, `2*K*BN^2` per thread
* Compute / IO: `BN*T`
* Concurrent threads in a threadgroup: `TGMem / (2*T*BN*BK*sizeof(float)`

### Optimized code


|Method                    | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |  notes |
|--------------------------|-------------|---------------|---------------|---------------|--------|
|merge duplicated compute  |     100     |      1817     |      2492     |      2389     |  |
|read cache from registry  |     104     |      1833     |      2605     |      2399     |  |


Surprisingly, moving `ac_ptr`, `bc_ptr` computation out of for-loop in the version without registry will significantly reduce performance.




## Acknowledge

* https://github.com/siboehm/SGEMM_CUDA
* https://github.com/bkvogel/metal_performance_testing
* https://developer.apple.com/forums/thread/105534
