import cxxCUBLAS

extension cublasOperation {
    public var ascublas: cublasOperation_t {
        return .init(UInt32(self.rawValue))
    }
}

extension cublasComputeType {
    public var ascublas: cublasComputeType_t {
        return .init(UInt32(self.rawValue))
    }
}

extension cublasGemmAlgo {
    public var ascublas: cublasGemmAlgo_t {
        return .init(Int32(self.rawValue))
    }
}
