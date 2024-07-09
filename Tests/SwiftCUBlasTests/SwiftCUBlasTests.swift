import SwiftCU
import XCTest
import cxxCUBlas
import cxxCU
import PythonKit
import SwiftCUBlas
let npy = Python.import("numpy")

@testable import SwiftCUBlas

final class SwiftCUBlasTests: XCTestCase {
    func testSimpleMatmulRowMajor() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Float32] = [1.0, 2.0, 3.0, 4.0,
                          5.0, 6.0, 7.0, 8.0]

        var B: [Float32] = [8.0, 7.0,
                        6.0, 5.0,
                        4.0, 3.0,
                        2.0, 1.0]

        var C: [Float32] = [Float32](repeating: 0.0, count: m * n)

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?
        defer {
            _ = aPointer.cudaAndHostDeallocate()
            _ = bPointer.cudaAndHostDeallocate()
            _ = cPointer.cudaAndHostDeallocate()
        }
        let f32Size =  MemoryLayout<Float32>.stride
        _ = aPointer.cudaMemoryAllocate(m * k * f32Size)
        _ = bPointer.cudaMemoryAllocate(k * n * f32Size)
        _ = cPointer.cudaMemoryAllocate(m * n * f32Size)

        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * f32Size, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * f32Size, copyKind: .cudaMemcpyHostToDevice)

        let handle = CUBlasHandle()
        var params = CUBlasParams<Float32>(
            fromRowMajor: aPointer!.assumingMemoryBound(to: Float32.self), B: bPointer!.assumingMemoryBound(to: Float32.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: 1.0, beta: 0.0
        )

        let status = handle.sgemm_v2(params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes  { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let expectedC = matMulRowMajor(A: A, B: B, m: m, n: n, k: k)
        XCTAssert((0..<expectedC.count).allSatisfy { expectedC[$0] == C[$0] })
        let npyMatmul: [Float32] = Array(npy.matmul(npy.array(A).reshape([2,4]), npy.array(B).reshape([4,2])).flatten())!
        XCTAssert((0..<expectedC.count).allSatisfy { npyMatmul[$0] == C[$0] })
    }

    func testSimpleMatmulColumnMajor() throws {
        let m = 2
        let n = 2
        let k = 4

        var A: [Float32] = [1.0, 5.0,
                            2.0, 6.0,
                            3.0, 7.0,
                            4.0, 8.0]

        var B: [Float32] = [8.0, 6.0, 4.0, 2.0,
                            7.0, 5.0, 3.0, 1.0]

        var C: [Float32] = [Float32](repeating: 0.0, count: m * n)

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?
        defer {
            _ = aPointer.cudaAndHostDeallocate()
            _ = bPointer.cudaAndHostDeallocate()
            _ = cPointer.cudaAndHostDeallocate()
        }
        let f32Size =  MemoryLayout<Float32>.stride
        _ = aPointer.cudaMemoryAllocate(m * k * f32Size)
        _ = bPointer.cudaMemoryAllocate(k * n * f32Size)
        _ = cPointer.cudaMemoryAllocate(m * n * f32Size)

        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * f32Size, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * f32Size, copyKind: .cudaMemcpyHostToDevice)

        let handle = CUBlasHandle()
        var params = CUBlasParams<Float32>(
            fromColumnMajor: aPointer!.assumingMemoryBound(to: Float32.self), B: bPointer!.assumingMemoryBound(to: Float32.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), m: Int32(m), n: Int32(n), k: Int32(k), alpha: 1.0, beta: 0.0
        )

        let status = handle.sgemm_v2(params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes  { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(fromMutableRawPointer: cPointer, numberOfBytes: m * n * f32Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let npyMatmul = npy.matmul(npy.array(A).reshape([2,4], order: "F"), npy.array(B).reshape([4,2], order: "F"))
        let cNpyArray = npy.array(C).reshape([2,2], order: "F")
        XCTAssert(Bool(npy.allclose(npyMatmul, cNpyArray))!)
    }
}
