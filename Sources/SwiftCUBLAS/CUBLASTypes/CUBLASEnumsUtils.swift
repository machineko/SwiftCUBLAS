import cxxCUBLAS

extension cublasOperation {
    var ascublas : cublasOperation_t {
        return .init(UInt32(self.rawValue))
    }
}