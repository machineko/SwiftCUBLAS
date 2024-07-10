import cxxCUBLAS

public protocol CUBLASDataType {}

extension Float16: CUBLASDataType {}
extension Float: CUBLASDataType {}
extension Double: CUBLASDataType {}
extension UInt8: CUBLASDataType {}
extension Int32: CUBLASDataType {}
extension __half: CUBLASDataType {}

public struct CUBLASHandle: ~Copyable {
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

public struct CUBLASParams<T: CUBLASDataType>: ~Copyable {
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

public struct CUBLASParamsMixed<inputType: CUBLASDataType, outputType: CUBLASDataType, computeType>: ~Copyable {
    public var A: UnsafePointer<inputType>
    public var B: UnsafePointer<inputType>
    public var C: UnsafeMutablePointer<outputType>
    public let m: Int32, n: Int32, k: Int32, lda: Int32, ldb: Int32, ldc: Int32, batchSize: Int32
    public var alpha: computeType, beta: computeType
    
    deinit {
        A.deallocate()
        B.deallocate()
        C.deallocate()
    }
}