public enum cublasStatus: Int {
    case cublas_status_success = 0
    case cublas_status_not_initialized = 1
    case cublas_status_alloc_failed = 3
    case cublas_status_invalid_value = 7
    case cublas_status_arch_mismatch = 8
    case cublas_status_mapping_error = 11
    case cublas_status_execution_failed = 13
    case cublas_status_internal_error = 14
    case cublas_status_not_supported = 15
    case cublas_status_license_error = 16
}

public enum cublasFillMode: Int {
    case cublas_fill_mode_lower = 0
}

public enum cublasDiagType: Int {
    case cublas_diag_non_unit = 0
}

public enum cublasSideMode: Int {
    case cublas_side_left = 0
}

public enum cublasOperation: Int {
    case cublas_op_n = 0
    case cublas_op_t = 1
    case cublas_op_c = 2
    static var cublas_op_hermitan: cublasOperation { return .cublas_op_c }  // synonym if CUBLAS_OP_C
    case cublas_op_conjg = 3  // conjugate, placeholder - not supported in the current release
}

public enum cublasPointerMode: Int {
    case cublas_pointer_mode_host = 0
}

public enum cublasAtomicsMode: Int {
    case cublas_atomics_not_allowed = 0
}

public enum cublasGemmAlgo: Int {
    case cublas_gemm_dfalt = -1
    static var cublas_gemm_default: cublasGemmAlgo { return .cublas_gemm_dfalt }
    case cublas_gemm_algo0 = 0
    case cublas_gemm_algo1 = 1
    case cublas_gemm_algo2 = 2
    case cublas_gemm_algo3 = 3
    case cublas_gemm_algo4 = 4
    case cublas_gemm_algo5 = 5
    case cublas_gemm_algo6 = 6
    case cublas_gemm_algo7 = 7
    case cublas_gemm_algo8 = 8
    case cublas_gemm_algo9 = 9
    case cublas_gemm_algo10 = 10
    case cublas_gemm_algo11 = 11
    case cublas_gemm_algo12 = 12
    case cublas_gemm_algo13 = 13
    case cublas_gemm_algo14 = 14
    case cublas_gemm_algo15 = 15
    case cublas_gemm_algo16 = 16
    case cublas_gemm_algo17 = 17
    case cublas_gemm_algo18 = 18
    case cublas_gemm_algo19 = 19
    case cublas_gemm_algo20 = 20
    case cublas_gemm_algo21 = 21
    case cublas_gemm_algo22 = 22
    case cublas_gemm_algo23 = 23
    case cublas_gemm_default_tensor_op = 99
    static var cublas_gemm_dfalt_tensor_op: cublasGemmAlgo { return .cublas_gemm_default_tensor_op }
    case cublas_gemm_algo0_tensor_op = 100
    case cublas_gemm_algo1_tensor_op = 101
    case cublas_gemm_algo2_tensor_op = 102
    case cublas_gemm_algo3_tensor_op = 103
    case cublas_gemm_algo4_tensor_op = 104
    case cublas_gemm_algo5_tensor_op = 105
    case cublas_gemm_algo6_tensor_op = 106
    case cublas_gemm_algo7_tensor_op = 107
    case cublas_gemm_algo8_tensor_op = 108
    case cublas_gemm_algo9_tensor_op = 109
    case cublas_gemm_algo10_tensor_op = 110
    case cublas_gemm_algo11_tensor_op = 111
    case cublas_gemm_algo12_tensor_op = 112
    case cublas_gemm_algo13_tensor_op = 113
    case cublas_gemm_algo14_tensor_op = 114
    case cublas_gemm_algo15_tensor_op = 115
}

public enum cublasMath: Int {
    case cublas_default_math = 0
    case cublas_tensor_op_math = 1
    case cublas_pedantic_math = 2
    case cublas_tf32_tensor_op_math = 3
    case cublas_math_disallow_reduced_precision_reduction = 16
}

public enum cublasComputeType: Int {
    case cublas_compute_16f = 64  // half - default
    case cublas_compute_16f_pedantic = 65  // half - pedantic
    case cublas_compute_32f = 68  // float - default
    case cublas_compute_32f_pedantic = 69  // float - pedantic
    case cublas_compute_32f_fast_16f = 74  // float - fast, allows down-converting inputs to half or TF32
    case cublas_compute_32f_fast_16bf = 75  // float - fast, allows down-converting inputs to bfloat16 or TF32
    case cublas_compute_32f_fast_tf32 = 77  // float - fast, allows down-converting inputs to TF32
    case cublas_compute_64f = 70  // double - default
    case cublas_compute_64f_pedantic = 71  // double - pedantic
    case cublas_compute_32i = 72  // signed 32-bit int - default
    case cublas_compute_32i_pedantic = 73  // signed 32-bit int - pedantic
}
