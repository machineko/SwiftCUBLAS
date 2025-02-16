// swift-tools-version: 6.0
import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath: String = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
    let cuLibPath = "-L\(cuPath)\\lib\\x64"
    let cuIncludePath = "-I\(cuPath)\\include"
    
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuLibPath = "-L\(cuPath)/lib64"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif

let package = Package(
    name: "SwiftCUBLAS",
    products: [
        .library(
            name: "SwiftCUBLAS",
            targets: ["SwiftCUBLAS"]),
        .library(
            name: "cxxCUBLAS",
            targets: ["cxxCUBLAS"]),
    ],
    dependencies: 
    [
        .package(url: "https://github.com/machineko/SwiftCU", branch: "main")
    ],
    targets: [
        .target(
            name: "cxxCUBLAS",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath(cuIncludePath)
            ],
            linkerSettings: [
                .unsafeFlags([
                    cuLibPath,
                ]),
                .linkedLibrary("cublas"),
            ]
        ),
        .target(
            name: "SwiftCUBLAS",
            dependencies: [
                "cxxCUBLAS",
                .product(name: "SwiftCU", package: "SwiftCU"),
                .product(name: "cxxCU", package: "SwiftCU"),

            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(
                    [cuIncludePath]
                )
            ]
        ),
          .testTarget(
            name: "SwiftCUBLASTests",
           
            dependencies: [
                "cxxCUBLAS", "SwiftCUBLAS",
                .product(name: "SwiftCU", package: "SwiftCU"),
                .product(name: "cxxCU", package: "SwiftCU"),
            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)