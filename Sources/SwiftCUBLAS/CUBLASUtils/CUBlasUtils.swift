import Foundation
import SwiftCU
import cxxCUBLAS

extension cublasStatus_t {
    /// Converts the `cublasStatus_t` to a Swift `cublasStatus`.
    public var asSwift: cublasStatus {
        return cublasStatus(rawValue: Int(self.rawValue))!
    }
}

extension cublasStatus {
    /// Checks if the error represents a successful CUBLAS operation.
    public var isSuccessful: Bool {
        return self == .cublas_status_success
    }

    /// Checks the condition and throws a precondition failure if the error is not successful.
    /// - Parameter message: The message to include in the precondition failure.
    @inline(__always)
    public func safetyCheckCondition(message: String) {
        precondition(self.isSuccessful, "\(message): cudaErrorValue: \(self)")
    }
}

extension CUBLASParamsMixed {
    /// Gets the CUDA data type for the input type.
    var inputCUDAType: cudaDataType {
        switch inputType.self {
        case is Float.Type:
            return CUDA_R_32F
        case is Double.Type:
            return CUDA_R_64F
        case is Int8.Type:
            return CUDA_R_8I
        case is Int32.Type:
            return CUDA_R_32I
        case is Float16.Type:
            return CUDA_R_16F
        default:
            fatalError("\(inputType.self) not supported")
        }
    }

    /// Gets the CUDA data type for the output type.
    var outputCUDAType: cudaDataType {
        switch outputType.self {
        case is Float.Type:
            return CUDA_R_32F
        case is Double.Type:
            return CUDA_R_64F
        case is Int8.Type:
            return CUDA_R_8I
        case is Int32.Type:
            return CUDA_R_32I
        case is Float16.Type:
            return CUDA_R_16F
        default:
            fatalError("\(inputType.self) not supported")
        }
    }
}

extension CUBLASParamsMixed {
    /// Initializes the parameters from row-major data.
    /// - Parameters:
    ///   - A: Pointer to the first matrix.
    ///   - B: Pointer to the second matrix.
    ///   - C: Pointer to the result matrix.
    ///   - m: Number of rows of matrix C and matrix A.
    ///   - n: Number of columns of matrix C and matrix B.
    ///   - k: Number of columns of matrix A and rows of matrix B.
    ///   - batchSize: The batch size.
    ///   - alpha: Scalar multiplier for the product of matrices A and B.
    ///   - beta: Scalar multiplier for the matrix C.
    public init(
        fromRowMajor A: UnsafePointer<inputType>, B: UnsafePointer<inputType>, C: UnsafeMutablePointer<outputType>, m: Int32, n: Int32,
        k: Int32, batchSize: Int32 = 0, alpha: computeType, beta: computeType
    ) {
        // cublas is using column-major memory order; swapping B with A and n -> m will result in row-major results
        self.A = B
        self.B = A
        self.C = C

        self.alpha = alpha
        self.beta = beta

        self.m = n
        self.n = m
        self.k = k
        self.batchSize = batchSize

        self.lda = n
        self.ldb = k
        self.ldc = n
    }

    /// Initializes the parameters from column-major data.
    /// - Parameters:
    ///   - A: Pointer to the first matrix.
    ///   - B: Pointer to the second matrix.
    ///   - C: Pointer to the result matrix.
    ///   - m: Number of rows of matrix C and matrix A.
    ///   - n: Number of columns of matrix C and matrix B.
    ///   - k: Number of columns of matrix A and rows of matrix B.
    ///   - batchSize: The batch size.
    ///   - alpha: Scalar multiplier for the product of matrices A and B.
    ///   - beta: Scalar multiplier for the matrix C.
    public init(
        fromColumnMajor A: UnsafePointer<inputType>, B: UnsafePointer<inputType>, C: UnsafeMutablePointer<outputType>, m: Int32, n: Int32,
        k: Int32, batchSize: Int32 = 0, alpha: computeType, beta: computeType
    ) {
        self.A = A
        self.B = B
        self.C = C

        self.alpha = alpha
        self.beta = beta

        self.m = m
        self.n = n
        self.k = k
        self.batchSize = batchSize

        self.lda = m
        self.ldb = k
        self.ldc = m
    }
}

extension CUBLASParams {
    /// Initializes the parameters from row-major data.
    /// - Parameters:
    ///   - A: Pointer to the first matrix.
    ///   - B: Pointer to the second matrix.
    ///   - C: Pointer to the result matrix.
    ///   - m: Number of rows of matrix C and matrix A.
    ///   - n: Number of columns of matrix C and matrix B.
    ///   - k: Number of columns of matrix A and rows of matrix B.
    ///   - batchSize: The batch size.
    ///   - alpha: Scalar multiplier for the product of matrices A and B.
    ///   - beta: Scalar multiplier for the matrix C.
    public init(
        fromRowMajor A: UnsafePointer<T>, B: UnsafePointer<T>, C: UnsafeMutablePointer<T>, m: Int32, n: Int32, k: Int32,
        batchSize: Int32 = 0, alpha: T, beta: T
    ) where T: CUBLASDataType {
        // cublas is using column-major memory order; swapping B with A and n -> m will result in row-major results
        self.A = B
        self.B = A
        self.C = C

        self.alpha = alpha
        self.beta = beta

        self.m = n
        self.n = m
        self.k = k
        self.batchSize = batchSize

        self.lda = n
        self.ldb = k
        self.ldc = n
    }

    /// Initializes the parameters from column-major data.
    /// - Parameters:
    ///   - A: Pointer to the first matrix.
    ///   - B: Pointer to the second matrix.
    ///   - C: Pointer to the result matrix.
    ///   - m: Number of rows of matrix C and matrix A.
    ///   - n: Number of columns of matrix C and matrix B.
    ///   - k: Number of columns of matrix A and rows of matrix B.
    ///   - batchSize: The batch size.
    ///   - alpha: Scalar multiplier for the product of matrices A and B.
    ///   - beta: Scalar multiplier for the matrix C.
    public init(
        fromColumnMajor A: UnsafePointer<T>, B: UnsafePointer<T>, C: UnsafeMutablePointer<T>, m: Int32, n: Int32, k: Int32,
        batchSize: Int32 = 0, alpha: T, beta: T
    ) where T: CUBLASDataType {
        self.A = A
        self.B = B
        self.C = C

        self.alpha = alpha
        self.beta = beta

        self.m = m
        self.n = n
        self.k = k
        self.batchSize = batchSize

        self.lda = m
        self.ldb = k
        self.ldc = m
    }
}

extension CUBLASHandle {
    /// Initializes a new CUBLAS handle with a CUDA stream.
    /// - Parameter stream: The CUDA stream to associate with the handle.
    public init(stream: inout cudaStream) {
        var handle: cublasHandle_t?
        let status = cublasCreate_v2(&handle).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't create handler cublasError")
        #endif
        let streamSetStatus = cublasSetStream_v2(handle, stream.stream).asSwift
        #if safetyCheck
            streamSetStatus.safetyCheckCondition(message: "Can't set stream on handler cublasError")
        #endif
        self.handle = handle
    }

    /// Performs single-precision general matrix multiplication (SGEMM) using CUBLAS.
    /// - Parameters:
    ///   - transposeA: Specifies whether to transpose matrix A.
    ///   - transposeB: Specifies whether to transpose matrix B.
    ///   - params: The parameters for the SGEMM operation.
    /// - Returns: The status of the SGEMM operation.
    public func sgemm_v2(
        transposeA: cublasOperation = .cublas_op_n, transposeB: cublasOperation = .cublas_op_n, params: inout CUBLASParams<Float32>
    ) -> cublasStatus {
        let status = cublasSgemm_v2(
            self.handle, transposeA.ascublas, transposeB.ascublas, params.m, params.n,
            params.k, &params.alpha, params.A, params.lda, params.B, params.ldb, &params.beta, params.C, params.ldc
        ).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't run sgemm cublasSgemm_v2 function \(status)")
        #endif
        return status
    }

    /// Performs mixed-precision general matrix multiplication (GEMM) using CUBLAS.
    /// - Parameters:
    ///   - transposeA: Specifies whether to transpose matrix A.
    ///   - transposeB: Specifies whether to transpose matrix B.
    ///   - params: The parameters for the GEMM operation.
    ///   - computeType: The compute type for the GEMM operation.
    ///   - cublasGemmAlgo: The algorithm to use for the GEMM operation.
    /// - Returns: The status of the GEMM operation.
    public func gemmEx<inputType: CUBLASDataType, outputType: CUBLASDataType, computeType: CUBLASDataType>(
        transposeA: cublasOperation = .cublas_op_n,
        transposeB: cublasOperation = .cublas_op_n,
        params: inout CUBLASParamsMixed<inputType, outputType, computeType>,
        computeType: cublasComputeType = .cublas_compute_16f, cublasGemmAlgo: cublasGemmAlgo = .cublas_gemm_default
    ) -> cublasStatus {
        let status = cublasGemmEx(
            self.handle,
            transposeA.ascublas,
            transposeB.ascublas,
            params.m, params.n, params.k,
            &params.alpha,
            params.A, params.inputCUDAType, params.lda,
            params.B, params.inputCUDAType, params.ldb,
            &params.beta,
            params.C, params.outputCUDAType, params.ldc,
            computeType.ascublas,
            cublasGemmAlgo.ascublas
        ).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't run cublasGemmEx function \(status)")
        #endif
        return status
    }
}
