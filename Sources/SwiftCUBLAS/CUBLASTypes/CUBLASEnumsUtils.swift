import cxxCUBLAS

public extension cublasOperation {
    var ascublas: cublasOperation_t {
        #if os(Linux)
            return .init(UInt32(self.rawValue))
        #elseif os(Windows)
            return .init(Int32(self.rawValue))
        #else
            fatalerror()
        #endif
    }
}

public extension cublasComputeType {
    var ascublas: cublasComputeType_t {
        #if os(Linux)
            return .init(UInt32(self.rawValue))
        #elseif os(Windows)
            return .init(Int32(self.rawValue))
        #else
            fatalerror()
        #endif
    }
}

public extension cublasGemmAlgo {
    var ascublas: cublasGemmAlgo_t {
        return .init(Int32(self.rawValue))
    }
}
