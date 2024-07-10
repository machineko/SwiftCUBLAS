import Foundation
import cxxCUBLAS
import SwiftCU

public extension cublasStatus_t {
    /// Converts the `cublasStatus_t` to a Swift `cublasStatus`.
    var asSwift: cublasStatus {
        return cublasStatus(rawValue: Int(self.rawValue))!
    }
}

public extension cublasStatus {
    /// Checks if the error represents a successful cublas operation.
    var isSuccessful: Bool {
        return self == .cublas_status_success
    }

    /// Checks the condition and throws a precondition failure if the error is not successful.
    /// - Parameter message: The message to include in the precondition failure.
    @inline(__always)
    func safetyCheckCondition(message: String) {
        precondition(self.isSuccessful, "\(message): cudaErrorValue: \(self)")
    }
}

extension CUBLASParamsMixed {
    var inputCUDAType: cudaDataType {
        switch inputType.self {
        case is Float.Type:
            return CUDA_R_32F
        case is Double.Type:
            return CUDA_R_64F
        case is UInt8.Type:
            return CUDA_R_8U
        case is Int32.Type:
            return CUDA_R_32I
        case is __half.Type:
            return CUDA_R_16F
        default:
            fatalError("Unsupported CUBLAS data type")
        }
    }

    var outputCUDAType: cudaDataType {
        switch outputType.self {
        case is Float.Type:
            return CUDA_R_32F
        case is Double.Type:
            return CUDA_R_64F
        case is UInt8.Type:
            return CUDA_R_8U
        case is Int32.Type:
            return CUDA_R_32I
        case is __half.Type:
            return CUDA_R_16F
        default:
            fatalError("Unsupported CUBLAS data type")
        }
    }
}

public extension CUBLASParamsMixed {
    init(fromRowMajor A: UnsafePointer<inputType>, B: UnsafePointer<inputType>, C: UnsafeMutablePointer<outputType>, m: Int32, n: Int32, k: Int32, batchSize: Int32 = 0, alpha: computeType, beta: computeType) {
        // cublas is using column-major memory order swapping B with A and n -> m will result in row-major results
        self.A = B
        self.B = A
        self.C = C

        self.alpha = alpha
        self.beta = beta

        // m -> Number of rows of matrix C and matrix A.
        // n -> Number of columns of matrix C and matrix B.
        // k -> Number of columns of matrix A and rows of matrix B.
        self.m = n
        self.n = m
        self.k = k
        self.batchSize = batchSize

        self.lda = n
        self.ldb = k
        self.ldc = n
    }

    // Initializer for column-major data
    init(fromColumnMajor A: UnsafePointer<inputType>, B: UnsafePointer<inputType>, C: UnsafeMutablePointer<outputType>, m: Int32, n: Int32, k: Int32, batchSize: Int32 = 0, alpha: computeType, beta: computeType) {
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

public extension CUBLASParams {
    init(fromRowMajor A: UnsafePointer<T>, B: UnsafePointer<T>, C: UnsafeMutablePointer<T>, m: Int32, n: Int32, k: Int32, batchSize: Int32 = 0, alpha: T, beta: T) where T: CUBLASDataType{
        // cublas is using column-major memory order swapping B with A and n -> m will result in row-major results
        self.A = B
        self.B = A
        self.C = C
        
        self.alpha = alpha
        self.beta = beta

        // m -> Number of rows of matrix C and matrix A.
        // n -> Number of columns of matrix C and matrix B.
        // k -> Number of columns of matrix A and rows of matrix B.
        self.m = n
        self.n = m
        self.k = k
        self.batchSize = batchSize

        self.lda = n
        self.ldb = k
        self.ldc = n
    }

    init(fromColumnMajor A: UnsafePointer<T>, B: UnsafePointer<T>, C: UnsafeMutablePointer<T>, m: Int32, n: Int32, k: Int32, batchSize: Int32 = 0, alpha: T, beta: T) where T: CUBLASDataType {
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

public extension CUBLASHandle {
    init(stream: inout cudaStream) {
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

    func sgemm_v2(transposeA: cublasOperation = .cublas_op_n, transposeB: cublasOperation = .cublas_op_n, params: inout CUBLASParams<Float32>) -> cublasStatus {
        let status = cublasSgemm_v2(
            self.handle, transposeA.ascublas, transposeB.ascublas, params.m, params.n, 
            params.k, &params.alpha, params.A, params.lda, params.B, params.ldb, &params.beta, params.C, params.ldc
        )
        return status.asSwift
    }

    func hgemm(transposeA: cublasOperation = .cublas_op_n, transposeB: cublasOperation = .cublas_op_n, params: inout CUBLASParams<__half>) -> cublasStatus {
        let status = cublasHgemm(
            self.handle, transposeA.ascublas, transposeB.ascublas, params.m, params.n, 
            params.k, &params.alpha, params.A, params.lda, params.B, params.ldb, &params.beta, params.C, params.ldc
        )
        return status.asSwift
    }

   func gemmEx(
        transposeA: cublasOperation = .cublas_op_n,
        transposeB: cublasOperation = .cublas_op_n,
        params: inout CUBLASParamsMixed<some CUBLASDataType, some CUBLASDataType, some CUBLASDataType>,
        computeType: cublasComputeType = .cublas_compute_16f, cublasGemmAlgo: cublasGemmAlgo = .cublas_gemm_dfalt
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
        )
        print(status.asSwift)
        return status.asSwift
    }

}

