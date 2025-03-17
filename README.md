# metal-matmul

This is step-by-step optimization and performance benchmark for matmul on metal.


## Instruction


```
cmake .. && make && ./benchmark
```


## Stats

Device: M1 Max, 64GB, OS 15.3.2
Performance (GFlops)

|Method      | f32 256x256 | f32 1024x1024 | f32 4096x4096 | f32 8192x8192 |
|------------|-------------|---------------|---------------|---------------|
|MPS (Swift) | 20          |  1262         | 7441          |  7428         |
|MPS (C++)   | 96.93       |  181.10       | 73.60         |  100.76       |
|naive       | 45.07       |  38.09        | 26.18         |  13.44        |
|naive 16x16 | 81.47       |  93.15        | 90.62         |  79.51        |


## Acknowledge

* https://github.com/siboehm/SGEMM_CUDA
* https://github.com/bkvogel/metal_performance_testing
* https://developer.apple.com/forums/thread/105534
