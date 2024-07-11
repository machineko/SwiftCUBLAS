# SwiftCUBLAS

SwiftCUBLAS is a wrapper for the cuBLAS library, providing GPU-accelerated linear algebra operations in Swift. It includes utilities for matrix operations and a robust suite of tests. The package is tested on the latest CUDA runtime API (v12.5) on both Linux and Windows.

| Operating System | Swift Version | CUDA Version | Supported |
|------------------|---------------|--------------|-----------|
| Linux            | 6.0           | 12.*         | ✅        |
| Windows 11       | 6.0           | 12.*         | ✅        |

## Installation

To include SwiftCUBLAS in your Swift project, add the following line to your `Package.swift` file:

```swift
.package(url: "https://github.com/machineko/SwiftCUBLAS", branch: "main")
```

## Documentation
Docc generated for Swift wrapped API [SwiftCUBLAS](https://swiftcublas.kobus.me/documentation/swiftcublas/cublashandle/)

CUDA runtime [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api)

## Example

```swift
@Test func testSimpleSGEMMRowMajor() async throws {
    let cuStatus = CUDevice(index: 0).setDevice()
    #expect(cuStatus)
    let m = 2
    let n = 2
    let k = 4

    var A: [Float32] = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]

    var B: [Float32] = [
        8.0, 7.0,
        6.0, 5.0,
        4.0, 3.0,
        2.0, 1.0,
    ]

    var C: [Float32] = [Float32](repeating: 0.0, count: m * n)

    var aPointer: UnsafeMutableRawPointer?
    var bPointer: UnsafeMutableRawPointer?
    var cPointer: UnsafeMutableRawPointer?
    defer {
        _ = aPointer.cudaAndHostDeallocate()
        _ = bPointer.cudaAndHostDeallocate()
        _ = cPointer.cudaAndHostDeallocate()
    }
    let f32Size = MemoryLayout<Float32>.stride
    _ = aPointer.cudaMemoryAllocate(m * k * f32Size)
    _ = bPointer.cudaMemoryAllocate(k * n * f32Size)
    _ = cPointer.cudaMemoryAllocate(m * n * f32Size)

    _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * f32Size, copyKind: .cudaMemcpyHostToDevice)
    _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * f32Size, copyKind: .cudaMemcpyHostToDevice)

    let handle = CUBLASHandle()
    var params = CUBLASParams<Float32>(
        fromRowMajor: aPointer!.assumingMemoryBound(to: Float32.self), B: bPointer!.assumingMemoryBound(to: Float32.self),
        C: cPointer!.assumingMemoryBound(to: Float32.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: 1.0, beta: 0.0
    )

    let status = handle.sgemm_v2(params: &params)
    #expect(status.isSuccessful)
    C.withUnsafeMutableBytes { rawBufferPointer in
        var pointerAddress = rawBufferPointer.baseAddress
        let outStatus = pointerAddress.cudaMemoryCopy(
            fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
        #expect(outStatus.isSuccessful)
    }
    cudaDeviceSynchronize()
    let cExpected = matrixMultiply(m, n, k, A, B, isRowMajor: true)
    #expect(cExpected ~= C)
}
```
#### For more examples check test cases or cublas docs

# Testing

Current version of SwiftCUBLAS is tested on Swift 6.0 development branch using swift-testing package and CUDA v12.5