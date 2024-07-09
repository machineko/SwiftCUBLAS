import cxxCUBlas

public protocol CuBlasDataType {}

extension Float16: CuBlasDataType {}
extension Float: CuBlasDataType {}
extension Double: CuBlasDataType {}
extension Int8: CuBlasDataType {}
extension Int32: CuBlasDataType {}

public struct CUBlasHandle: ~Copyable {
    public var handle: cublasHandle_t?

    public init() {
        var handle: cublasHandle_t?
        let status = cublasCreate_v2(&handle).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't create handler cublasError: \(status)")
        #endif
        self.handle = handle
    }

    deinit {
        let status = cublasDestroy_v2(handle).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't launch kernel cublasError: \(status)")
        #endif
    }
}

public struct CUBlasParams<T: CuBlasDataType>: ~Copyable {
    public var A: UnsafePointer<T>
    public var B: UnsafePointer<T>
    public var C: UnsafeMutablePointer<T>
    public let m: Int32, n: Int32, k: Int32, lda: Int32, ldb: Int32, ldc: Int32, batchSize: Int32
    public var alpha: T, beta: T

    deinit {
        A.deallocate()
        B.deallocate()
        C.deallocate()
    }
}