import cxxCUBLAS

extension cublasOperation {
    public var ascublas: cublasOperation_t {
        #if os(Linux)
            return .init(UInt32(self.rawValue))
        #elseif os(Windows)
            return .init(Int32(self.rawValue))
        #else
            fatalerror()
        #endif
    }
}

extension cublasComputeType {
    public var ascublas: cublasComputeType_t {
        #if os(Linux)
            return .init(UInt32(self.rawValue))
        #elseif os(Windows)
            return .init(Int32(self.rawValue))
        #else
            fatalerror()
        #endif
    }
}

extension cublasGemmAlgo {
    public var ascublas: cublasGemmAlgo_t {
        return .init(Int32(self.rawValue))
    }
}
