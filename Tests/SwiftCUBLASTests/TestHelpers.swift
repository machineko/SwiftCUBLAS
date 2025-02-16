import SwiftCUBLAS

func getIndex(row: Int, col: Int, numRows: Int, numCols: Int, isRowMajor: Bool) -> Int {
    return isRowMajor ? row * numCols + col : col * numRows + row
}

func matrixMultiply<T: CUBLASDataType & Numeric>(
    _ m: Int,
    _ n: Int,
    _ k: Int,
    _ A: [T],
    _ B: [T],
    isRowMajor: Bool
) -> [T] {
    var C: [T] = [T](repeating: 0, count: m * n)
    for i in 0..<m {
        for j in 0..<n {
            var sum: T = 0
            for p in 0..<k {
                let aIndex = getIndex(row: i, col: p, numRows: m, numCols: k, isRowMajor: isRowMajor)
                let bIndex = getIndex(row: p, col: j, numRows: k, numCols: n, isRowMajor: isRowMajor)
                sum += A[aIndex] * B[bIndex]
            }
            let cIndex = getIndex(row: i, col: j, numRows: m, numCols: n, isRowMajor: isRowMajor)
            C[cIndex] = sum
        }
    }
    return C
}
