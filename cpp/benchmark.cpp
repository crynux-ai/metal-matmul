#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct RunParams {
    uint N;
    uint M;
    uint K;
};

void metal_matmul(uint N, uint M, uint K, const std::string& method,
        int repeat=1,
        const std::array<int, 2>& threads_per_group={1, 1},
        bool verify=false
) {
    @autoreleasepool {
        NSError* error;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Metal is not supported");
        }

        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"matmul" ofType:@"metallib"];
        id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&error];
        if (!library) {
            throw std::runtime_error("Fail to load library");
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:[NSString stringWithUTF8String:method.c_str()]];
        if (!kernelFunction) {
            throw std::runtime_error("Method not found in the library");
        }

        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!state) {
            throw std::runtime_error("Fail to create pipeline");
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        RunParams param = {
            .N = N,
            .M = M,
            .K = K,
        };

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::unique_ptr<float[]> A = std::make_unique<float[]>(N * K);
        std::unique_ptr<float[]> B = std::make_unique<float[]>(M * K);
        std::unique_ptr<float[]> C = std::make_unique<float[]>(N * M);
        std::generate(A.get(), A.get() + N * K, [&]() { return dis(gen); });
        std::generate(B.get(), B.get() + M * K, [&]() { return dis(gen); });

        id<MTLBuffer> bufferParam = [device newBufferWithBytes:&param length:sizeof(param) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A.get() length:N*K*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B.get() length:M*K*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithBytes:C.get() length:N*M*sizeof(float) options:MTLResourceStorageModeShared];


        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            [computeEncoder setComputePipelineState:state];
            [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
            [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
            [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
            [computeEncoder setBuffer:bufferC offset:0 atIndex:3];

            MTLSize gridSize = MTLSizeMake(N, M, 1);
            MTLSize threadgroupSize = MTLSizeMake(threads_per_group[0], threads_per_group[1], 1);
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        double gflops = 2 * double(N)*double(M)*double(K) / (double(duration.count()) / double(repeat));
        printf("%6dx%6d: %20s runtime: %16lld ns   %8.2fGFlops\n", N, M, method.c_str(), duration.count() / repeat, gflops);

        if (!verify) {
            return;
        }

        // Verify results
        float* ptrc = static_cast<float*>(bufferC.contents);
        memcpy(C.get(), ptrc, N * M * sizeof(float));

        int cnt = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float val = 0;
                int aptr = i*K;
                int bptr = j*K;
                for (int k = 0; k < K; k++, aptr++, bptr++) {
                    val += A[aptr] * B[bptr];
                }
                if (fabs(val - C[i * M + j]) < 1e-5) {
                    cnt++;
                } else {
                    printf("Mistmatch: %f - %f\n", val, C[i*M+j]);
                }
            }
        }
        printf("%d/%d match\n", cnt, N * M);
    }
}


int main() {
    metal_matmul(256, 256, 256, "naive", 1000, {1, 1}, false);
    metal_matmul(1024, 1024, 1024, "naive", 1000, {1, 1}, false);
    metal_matmul(4096, 4096, 4096, "naive", 10, {1, 1}, false);
    metal_matmul(8192, 8192, 8192, "naive", 3, {1, 1}, false);
}
