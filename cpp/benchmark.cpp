#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <format>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

struct RunParams {
    uint N;
    uint M;
    uint K;
};

id<MTLComputePipelineState> func_state(id<MTLDevice> device, const std::string& name) {
    NSError* error;
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"matmul" ofType:@"metallib"];
    id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&error];
    if (!library) {
        throw std::runtime_error("Fail to load library");
    }
    id<MTLFunction> kernelFunction = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!kernelFunction) {
        throw std::runtime_error("Method not found in the library");
    }
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!state) {
        throw std::runtime_error("Fail to create pipeline");
    }
    return state;
}

void mps(id<MTLCommandBuffer> commandBuffer, id<MTLDevice> device,
         id<MTLBuffer> bufferA, id<MTLBuffer> bufferB, id<MTLBuffer> bufferC,
         uint N, uint M, uint K) {
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:M rowBytes:M*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:M rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
        transposeLeft:NO transposeRight:NO resultRows:N resultColumns:M interiorColumns:K
        alpha:1.0 beta:0.0];
    MPSCommandBuffer *mpsCommandBuffer = [[MPSCommandBuffer alloc] initWithCommandBuffer:commandBuffer];
    [matmul encodeToCommandBuffer:mpsCommandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
}

void naive(id<MTLCommandBuffer> commandBuffer, id<MTLDevice> device,
           id<MTLBuffer> bufferParam, id<MTLBuffer> bufferA, id<MTLBuffer> bufferB, id<MTLBuffer> bufferC,
           const RunParams& param, const std::array<int, 2>& threads_per_group) {
    id<MTLComputePipelineState> state = func_state(device, "naive");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:state];
    [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:3];
    MTLSize gridSize = MTLSizeMake(param.N, param.M, 1);
    MTLSize threadgroupSize = MTLSizeMake(threads_per_group[0], threads_per_group[1], 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
}

void shared_mem(
        id<MTLCommandBuffer> commandBuffer, id<MTLDevice> device,
        id<MTLBuffer> bufferParam, id<MTLBuffer> bufferA, id<MTLBuffer> bufferB, id<MTLBuffer> bufferC,
        const RunParams& param, const std::array<int, 2>& threads_per_group) {
    id<MTLComputePipelineState> state = func_state(device, "shared_mem_4");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setThreadgroupMemoryLength:threads_per_group[0]*threads_per_group[1]*8 atIndex:0];
    [computeEncoder setComputePipelineState:state];
    [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:3];
    MTLSize gridSize = MTLSizeMake(param.N, param.M, 1);
    MTLSize threadgroupSize = MTLSizeMake(threads_per_group[0], threads_per_group[1], 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
}

void block_tiling(
        id<MTLCommandBuffer> commandBuffer, id<MTLDevice> device,
        id<MTLBuffer> bufferParam, id<MTLBuffer> bufferA, id<MTLBuffer> bufferB, id<MTLBuffer> bufferC,
        const RunParams& param, const std::array<int, 2>& threads_per_group) {
    int BLOCK_K = 4;
    int BLOCK_N = 4;
    assert(BLOCK_N * BLOCK_K >= threads_per_group[1]);
    id<MTLComputePipelineState> state = func_state(device, std::format("block_tiling_{}_{}", BLOCK_K, BLOCK_N));
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    int tg_bytes = 8*threads_per_group[0]*BLOCK_N*BLOCK_K;
    assert(tg_bytes < device.maxThreadgroupMemoryLength);
    [computeEncoder setThreadgroupMemoryLength:tg_bytes atIndex:0];
    [computeEncoder setComputePipelineState:state];
    [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:3];
    MTLSize gridSize = MTLSizeMake(param.N / BLOCK_N, param.M / BLOCK_N, 1);
    MTLSize threadgroupSize = MTLSizeMake(threads_per_group[0], threads_per_group[1], 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
}

void float4x4_kernel(
        id<MTLCommandBuffer> commandBuffer, id<MTLDevice> device,
        id<MTLBuffer> bufferParam, id<MTLBuffer> bufferA, id<MTLBuffer> bufferB, id<MTLBuffer> bufferC,
        const RunParams& param, const std::array<int, 2>& threads_per_group) {
    int BLOCK_K = 4;
    int BLOCK_N = 4;
    assert(BLOCK_N * BLOCK_K >= threads_per_group[1]);
    id<MTLComputePipelineState> state = func_state(device, "float4x4_kernel");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    int tg_bytes = 8*threads_per_group[0]*BLOCK_N*BLOCK_K;
    assert(tg_bytes < device.maxThreadgroupMemoryLength);
    [computeEncoder setThreadgroupMemoryLength:tg_bytes atIndex:0];
    [computeEncoder setComputePipelineState:state];
    [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:3];
    MTLSize gridSize = MTLSizeMake(param.N / BLOCK_N, param.M / BLOCK_N, 1);
    MTLSize threadgroupSize = MTLSizeMake(threads_per_group[0], threads_per_group[1], 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
}


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
        printf("Device info:\n\tMax threadgroup memory: %ld\n\tMax thread per group: %d, %d, %d\n",
            long(device.maxThreadgroupMemoryLength),
            int(device.maxThreadsPerThreadgroup.width),
            int(device.maxThreadsPerThreadgroup.height),
            int(device.maxThreadsPerThreadgroup.depth));

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

        size_t a_bytes = N*K*sizeof(float);
        size_t b_bytes = M*K*sizeof(float);
        size_t c_bytes = N*M*sizeof(float);

        id<MTLBuffer> bufferParam = [device newBufferWithBytes:&param length:sizeof(param) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A.get() length:a_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B.get() length:b_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithBytes:C.get() length:c_bytes options:MTLResourceStorageModeShared];


        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];


            if (method == "mps") {
                mps(commandBuffer, device, bufferA, bufferB, bufferC, N, M, K);
            } else if (method == "naive") {
                naive(commandBuffer, device, bufferParam, bufferA, bufferB, bufferC, param, threads_per_group);
            } else if (method == "shared_mem") {
                shared_mem(commandBuffer, device, bufferParam, bufferA, bufferB, bufferC, param, threads_per_group);
            } else if (method == "block_tiling") {
                block_tiling(commandBuffer, device, bufferParam, bufferA, bufferB, bufferC, param, threads_per_group);
            } else if (method == "float4x4_kernel") {
                float4x4_kernel(commandBuffer, device, bufferParam, bufferA, bufferB, bufferC, param, threads_per_group);(commandBuffer, device, bufferParam, bufferA, bufferB, bufferC, param, threads_per_group);
            }

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

        int correct = 0;
        int wrong = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float val = 0;
                int aptr = i*K;
                int bptr = j*K;
                for (int k = 0; k < K; k++, aptr++, bptr++) {
                    val += A[aptr] * B[bptr];
                }
                if (fabs(val - C[i * M + j]) < 1e-5) {
                    correct++;
                } else {
                    wrong++;
                    bool print_res = false;
                    if (print_res && wrong < 1000) {
                        printf("Mistmatch[%d,%d]: %f - %f\n", i, j, val, C[i*M+j]);
                    }
                }
            }
        }
        printf("%d/%d match\n", correct, N * M);
    }
}


int main() {
    std::string method = "float4x4_kernel";
    std::array<int, 2> threads_per_group = {16, 16};
    metal_matmul(256, 256, 256, method, 1000, threads_per_group, true);
    metal_matmul(1024, 1024, 1024, method, 100, threads_per_group, true);
    metal_matmul(4096, 4096, 4096, method, 10, threads_per_group, false);
    metal_matmul(8192, 8192, 8192, method, 3, threads_per_group, false);
}
