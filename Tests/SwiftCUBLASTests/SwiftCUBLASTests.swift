import PythonKit
import SwiftCU
import XCTest
import cxxCU
import cxxCUBLAS

@testable import SwiftCUBLAS

let npy = Python.import("numpy")

final class SwiftCUBLASTests: XCTestCase {
    func testSimpleMatmulRowMajor() throws {
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
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul: [Float32] = Array(npy.matmul(npy.array(A).reshape([2, 4]), npy.array(B).reshape([4, 2])).flatten())!
        XCTAssert((0..<npyMatmul.count).allSatisfy { npyMatmul[$0] == C[$0] })
    }

    func testSimpleMatmulColumnMajor() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Float32] = [
            1.0, 5.0,
            2.0, 6.0,
            3.0, 7.0,
            4.0, 8.0,
        ]

        var B: [Float32] = [
            8.0, 6.0, 4.0, 2.0,
            7.0, 5.0, 3.0, 1.0,
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
            fromColumnMajor: aPointer!.assumingMemoryBound(to: Float32.self), B: bPointer!.assumingMemoryBound(to: Float32.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: 1.0, beta: 0.0
        )

        let status = handle.sgemm_v2(params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul = npy.matmul(npy.array(A).reshape([2, 4], order: "F"), npy.array(B).reshape([4, 2], order: "F"))
        let cNpyArray = npy.array(C).reshape([2, 2], order: "F")
        XCTAssert(Bool(npy.allclose(npyMatmul, cNpyArray))!)
    }

    func testSimpleMatmulRowMajorHalf() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Float16] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]

        var B: [Float16] = [
            8.0, 7.0,
            6.0, 5.0,
            4.0, 3.0,
            2.0, 1.0,
        ]

        var C: [Float16] = [Float16](repeating: 0.0, count: m * n)

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?
        defer {
            _ = aPointer.cudaAndHostDeallocate()
            _ = bPointer.cudaAndHostDeallocate()
            _ = cPointer.cudaAndHostDeallocate()
        }
        let f16Size = MemoryLayout<Float16>.stride
        _ = aPointer.cudaMemoryAllocate(m * k * f16Size)
        _ = bPointer.cudaMemoryAllocate(k * n * f16Size)
        _ = cPointer.cudaMemoryAllocate(m * n * f16Size)

        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * f16Size, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * f16Size, copyKind: .cudaMemcpyHostToDevice)

        let handle = CUBLASHandle()
        var params = CUBLASParams<__half>(
            fromRowMajor: aPointer!.assumingMemoryBound(to: __half.self), B: bPointer!.assumingMemoryBound(to: __half.self),
            C: cPointer!.assumingMemoryBound(to: __half.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: __half(1.0), beta: __half(0.0)
        )

        let status = handle.hgemm(params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f16Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul = npy.matmul(npy.array(A.map { Float32($0) }).reshape([2, 4]), npy.array(B.map { Float32($0) }).reshape([4, 2]))
        let cNpyArray = npy.array(C.map { Float32($0) }).reshape([2, 2])
        XCTAssert(Bool(npy.allclose(npyMatmul, cNpyArray))!)
    }

    func testSimpleMatmulColumnMajorHalf() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Float16] = [
            1.0, 5.0,
            2.0, 6.0,
            3.0, 7.0,
            4.0, 8.0,
        ]

        var B: [Float16] = [
            8.0, 6.0, 4.0, 2.0,
            7.0, 5.0, 3.0, 1.0,
        ]

        var C: [Float16] = [Float16](repeating: 0.0, count: m * n)

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
        var params = CUBLASParams<__half>(
            fromColumnMajor: aPointer!.assumingMemoryBound(to: __half.self), B: bPointer!.assumingMemoryBound(to: __half.self),
            C: cPointer!.assumingMemoryBound(to: __half.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: __half(1.0), beta: __half(0.0)
        )

        let status = handle.hgemm(params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul = npy.matmul(
            npy.array(A.map { Float32($0) }).reshape([2, 4], order: "F"), npy.array(B.map { Float32($0) }).reshape([4, 2], order: "F"))
        let cNpyArray = npy.array(C.map { Float32($0) }).reshape([2, 2], order: "F")
        XCTAssert(Bool(npy.allclose(npyMatmul, cNpyArray))!)
    }

}

final class SwiftCUBLASGenericTests: XCTestCase {

    func testSimpleMatmulRowMajorHalfF32() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Float16] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]

        var B: [Float16] = [
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
        let f16Size = MemoryLayout<Float16>.stride
        let f32Size = MemoryLayout<Float32>.stride

        _ = aPointer.cudaMemoryAllocate(m * k * f16Size)
        _ = bPointer.cudaMemoryAllocate(k * n * f16Size)
        _ = cPointer.cudaMemoryAllocate(m * n * f32Size)

        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * f16Size, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * f16Size, copyKind: .cudaMemcpyHostToDevice)

        let handle = CUBLASHandle()
        // Input types => __half, Output type => F32, compute type => F32
        var params = CUBLASParamsMixed<__half, Float32, Float32>(
            fromRowMajor: aPointer!.assumingMemoryBound(to: __half.self), B: bPointer!.assumingMemoryBound(to: __half.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: 1.0, beta: 0.0
        )

        let status = handle.gemmEx(params: &params, computeType: .cublas_compute_32f_fast_16bf)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul = npy.matmul(npy.array(A.map { Float32($0) }).reshape([2, 4]), npy.array(B.map { Float32($0) }).reshape([4, 2]))
        let cNpyArray = npy.array(C).reshape([2, 2])
        XCTAssert(Bool(npy.allclose(npyMatmul, cNpyArray))!)
    }

    func testSimpleMatmulRowMajorI8F32() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Int8] = [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ]

        var B: [Int8] = [
            8, 7,
            6, 5,
            4, 3,
            2, 1,
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
        let i8Size = MemoryLayout<Int8>.stride
        let f32Size = MemoryLayout<Float32>.stride

        _ = aPointer.cudaMemoryAllocate(m * k * i8Size)
        _ = bPointer.cudaMemoryAllocate(k * n * i8Size)
        _ = cPointer.cudaMemoryAllocate(m * n * f32Size)

        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * i8Size, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * i8Size, copyKind: .cudaMemcpyHostToDevice)

        let handle = CUBLASHandle()
        // Input types => Int8, Output type => F32, compute type => F32
        var params = CUBLASParamsMixed<Int8, Float32, Float32>(
            fromRowMajor: aPointer!.assumingMemoryBound(to: Int8.self), B: bPointer!.assumingMemoryBound(to: Int8.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: 1.0, beta: 0.0
        )

        let status = handle.gemmEx(params: &params, computeType: .cublas_compute_32f)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul = npy.matmul(npy.array(A).reshape([2, 4]), npy.array(B).reshape([4, 2]))
        let cNpyArray = npy.array(C).reshape([2, 2])
        XCTAssert(Bool(npy.allclose(npyMatmul, cNpyArray))!)
    }
}
