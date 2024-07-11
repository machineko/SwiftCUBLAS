import cxxCUBLAS

/// A protocol that represents types compatible with CUBLAS operations.
public protocol CUBLASDataType {}

extension Float16: CUBLASDataType {}
extension Float: CUBLASDataType {}
extension Double: CUBLASDataType {}
extension Int8: CUBLASDataType {}
extension Int32: CUBLASDataType {}
extension __half: CUBLASDataType {}

/// A structure that manages a CUBLAS handle.
public struct CUBLASHandle: ~Copyable {
    /// The underlying CUBLAS handle.
    public var handle: cublasHandle_t?

    /// Initializes a new CUBLAS handle.
    public init() {
        var handle: cublasHandle_t?
        let status = cublasCreate_v2(&handle).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't create handler cublasError: \(status)")
        #endif
        self.handle = handle
    }

    /// Deinitializes the CUBLAS handle, releasing any associated resources.
    deinit {
        let status = cublasDestroy_v2(handle).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't launch kernel cublasError: \(status)")
        #endif
    }
}

/// A structure that holds parameters for CUBLAS operations with a single data type.
public struct CUBLASParams<T: CUBLASDataType>: ~Copyable {
    /// Pointer to the first matrix (A).
    public var A: UnsafePointer<T>
    /// Pointer to the second matrix (B).
    public var B: UnsafePointer<T>
    /// Pointer to the result matrix (C).
    public var C: UnsafeMutablePointer<T>
    /// Dimensions of the matrices and other parameters.
    public let m: Int32, n: Int32, k: Int32, lda: Int32, ldb: Int32, ldc: Int32, batchSize: Int32
    /// Scalar multiplier for the product of matrices A and B.
    public var alpha: T
    /// Scalar multiplier for the matrix C.
    public var beta: T

    deinit {
        A.deallocate()
        B.deallocate()
        C.deallocate()
    }
}

/// A structure that holds parameters for CUBLAS operations with mixed data types.
public struct CUBLASParamsMixed<inputType: CUBLASDataType, outputType: CUBLASDataType, computeType: CUBLASDataType>: ~Copyable {
    /// Pointer to the first matrix (A).
    public var A: UnsafePointer<inputType>
    /// Pointer to the second matrix (B).
    public var B: UnsafePointer<inputType>
    /// Pointer to the result matrix (C).
    public var C: UnsafeMutablePointer<outputType>
    /// Dimensions of the matrices and other parameters.
    public let m: Int32, n: Int32, k: Int32, lda: Int32, ldb: Int32, ldc: Int32, batchSize: Int32
    /// Scalar multiplier for the product of matrices A and B.
    public var alpha: computeType
    /// Scalar multiplier for the matrix C.
    public var beta: computeType

    deinit {
        A.deallocate()
        B.deallocate()
        C.deallocate()
    }
}
