# metal-matmul

This is step-by-step optimization and performance benchmark for matmul on metal.


## Instruction


```
cmake .. && make && ./benchmark
```


## Stats

Device: M1 Max, 64GB, OS 15.3.2
Performance (GFlops)

|Method   | 256x256 | 1024x1024 | 4096x4096 | 8192x8192 |
|---------|---------|-----------|-----------|-----------|
|naive    | 45.07   |  38.09    | 26.18     |  13.44    |



## Acknowledge

* https://github.com/siboehm/SGEMM_CUDA
* https://github.com/bkvogel/metal_performance_testing
