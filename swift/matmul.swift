// From https://developer.apple.com/forums/thread/105534
// swiftc -O matmul.swift && ./matmul

import Foundation
import MetalPerformanceShaders

func gflops(time: Double, size: Int) -> Double {
    let ops = 2.0 * pow(Double(size), 3)
    return ops / time / 1E9
}

func foo(_ N: Int) -> Double {
    // Prepare some data
    let rowsA = N
    let columnsA = N

    let a = UnsafeMutablePointer<Float>.allocate(capacity: rowsA * columnsA)
    let arrayA = UnsafeMutableBufferPointer(start: a, count: rowsA * columnsA)
    arrayA.assign(repeating: Float(1.0))

    // Get the device
    let device = MTLCreateSystemDefaultDevice()!
    

    // Build matrix on device
    let rowBytesA = columnsA * MemoryLayout<Float>.stride

    let bufferA = device.makeBuffer(bytes: arrayA.baseAddress!, length: rowsA * rowBytesA, options: [])!
    let descrA = MPSMatrixDescriptor(rows: rowsA, columns: columnsA, rowBytes: rowBytesA, dataType: .float32)
    let matrixA = MPSMatrix(buffer: bufferA, descriptor: descrA)

    let bufferC = device.makeBuffer(length: columnsA * rowBytesA, options: [])!
    let descrC = MPSMatrixDescriptor(rows: columnsA, columns: columnsA, rowBytes: rowBytesA, dataType: .float32)
    let matrixC = MPSMatrix(buffer: bufferC, descriptor: descrC)

    // Prepare multiplication
    let matMul = MPSMatrixMultiplication(device: device, resultRows: columnsA, resultColumns: columnsA, interiorColumns: rowsA)

    // Run multiplication
    let startTime = CFAbsoluteTimeGetCurrent()

    for _ in 1...100 {
        let commandBuffer = device.makeCommandQueue()!.makeCommandBuffer()!
        matMul.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixA, resultMatrix: matrixC)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) / 100

    return elapsed
}

// Uniform nubmers on logarithmic scale for testing, between 32 and 4096
let sizes: [Int] = [256, 1024, 4096, 8192]
print(sizes)

let result_gflops = sizes.map { (N) -> Double in
    let time = foo(N)
    return gflops(time: time, size: N)
}

print(result_gflops.map { Int($0) })
