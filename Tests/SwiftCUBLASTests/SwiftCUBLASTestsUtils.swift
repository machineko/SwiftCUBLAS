
func matMulRowMajor(A: [Float], B: [Float], m: Int, n: Int, k: Int) -> [Float] {
   var C = [Float](repeating: 0.0, count: m * n)

    for i in 0..<m {
        for j in 0..<n {
            for l in 0..<k {
                C[i * n + j] += A[i * k + l] * B[l * n + j]
            }
        }
    }
    return C
}