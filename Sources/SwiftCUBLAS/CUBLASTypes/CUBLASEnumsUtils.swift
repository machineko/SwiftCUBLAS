import cxxCUBLAS

public extension cublasOperation  {
    var ascublas : cublasOperation_t {
        return .init(UInt32(self.rawValue))
    }
}

public extension cublasComputeType {
    var ascublas : cublasComputeType_t {
        return .init(UInt32(self.rawValue))
    }
}

public extension cublasGemmAlgo {
    var ascublas : cublasGemmAlgo_t {
        return .init(Int32(self.rawValue))
    }
}